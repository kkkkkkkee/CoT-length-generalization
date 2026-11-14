"""Oracle-position interventions, exact per-query L1 controls, no training.

Outputs are teacher-forced full-clause trajectory diagnostics. Oracle rule
positions are supplied to the intervention only; answer labels score predictions.
Random controls are feasible random vertex directions on the causal simplex,
not uniformly random points on an L1 sphere or matched support-size edits.
"""
from pathlib import Path
import math
import torch
from data import Task, permute_predicates
from sparse_data import tensor_hash
from routing_model import load_routing_model
from routing_experiment import parameter_hash
from sparse_experiment import file_hash


def conditions():
    primary=[dict(name='original',kind='original'),dict(name='sham',kind='sham')]
    primary += [dict(name=f'leak_rule_{f}',kind='leak_rule',fraction=f) for f in (.25,.5)]
    primary += [dict(name=f'boost_{m}',kind='boost',mass=m) for m in (.9,.99,1.)]
    primary += [dict(name=f'ratio_{r}',kind='ratio',ratio=r) for r in (.25,.5,.75)]
    primary += [dict(name='boost_balance',kind='boost_balance',mass=.99,ratio=.5)]
    primary += [dict(name=f'grid_m{m}_r{r}',kind='grid',mass=m,ratio=r)
                for m in (.5,.9,.99,1.) for r in (.25,.5,.75)]
    controls=[dict(name=f'random_{c["name"]}_s{s}',kind='random',reference=c['name'],draw=s)
              for c in primary[2:11] for s in range(3)]
    return primary+controls


def indices(a, rules, queries):
    b,heads,j,t=a.shape
    if heads!=1: raise ValueError('one-head diagnostic only')
    def expand(x):
        x=torch.as_tensor(x,dtype=torch.long,device=a.device)
        if x.ndim==1: x=x[None].expand(b,-1)
        if x.shape!=(b,j): raise ValueError('invalid rule/query shape')
        return x[:,None,:,None]
    ri,qi=expand(rules),expand(queries)
    if ((ri<0)|(ri>=qi)|(qi>=t)).any(): raise ValueError('invalid causal rule/current-answer keys')
    visible=(torch.arange(t,device=a.device)[None,None,None,:]<=qi)
    return ri,qi,visible


def diagnostics(a, changed, rules, queries):
    a=a.double();z=changed.double();ri,qi,_=indices(a,rules,queries)
    ap=z.gather(-1,ri).squeeze(-1);ay=z.gather(-1,qi).squeeze(-1);m=ap+ay
    return dict(mass=m,rule_share=torch.where(m>0,ap/m.clamp_min(1e-300),.5),
                rule_weight=ap,state_weight=ay,gap=(ap-ay).abs(),
                l1=(z-a).abs().sum(-1),entropy=-(z*z.clamp_min(1e-300).log()).sum(-1))


def intervene(attention, rules, queries, condition):
    a=attention.double();ri,qi,visible=indices(a,rules,queries)
    if not torch.isfinite(a).all() or (a<0).any() or (a.masked_select(~visible)!=0).any():
        raise ValueError('invalid causal attention')
    if not torch.allclose(a.sum(-1),torch.ones_like(a[...,0]),atol=1e-6,rtol=0):
        raise ValueError('attention must be normalized')
    ap=a.gather(-1,ri).squeeze(-1);ay=a.gather(-1,qi).squeeze(-1)
    m=ap+ay;r=torch.where(m>0,ap/m.clamp_min(1e-300),.5)
    k=condition['kind']
    if k in ('original','sham'):return attention.clone()
    if k=='leak_rule':
        # Only true rule is a donor; all other predicates receive equal mass.
        count=ri.shape[-2]
        mask=(torch.arange(a.shape[-1],device=a.device)[None,None,None,:]<count).expand_as(a).clone()
        mask.scatter_(-1,ri,False)
        if (mask.sum(-1)==0).any():raise ValueError('requires distractor predicate')
        transfer=condition['fraction']*ap
        z=a+mask*transfer[...,None]/mask.sum(-1)[...,None]
        z.scatter_add_(-1,ri,-transfer[...,None])
    else:
        m1=m;r1=r
        if k in ('boost','boost_balance'):m1=torch.maximum(m,torch.full_like(m,condition['mass']))
        if k=='grid':m1=torch.full_like(m,condition['mass'])
        if k in ('ratio','grid','boost_balance'):r1=torch.full_like(r,condition['ratio'])
        if k not in ('boost','boost_balance','grid','ratio'):raise ValueError(k)
        bg=visible.expand_as(a).clone().scatter_(-1,ri,False).scatter_(-1,qi,False)
        total=(a*bg).sum(-1,keepdim=True);n=bg.sum(-1,keepdim=True)
        if ((n[...,0]==0)&(m1<1)).any():raise ValueError('no background support')
        prop=torch.where(total>0,a*bg/total.clamp_min(1e-300),bg/n.clamp_min(1))
        z=prop*(1-m1)[...,None]
        z.scatter_(-1,ri,(m1*r1)[...,None]);z.scatter_(-1,qi,(m1*(1-r1))[...,None])
    return z.to(attention.dtype)


def matched_random(attention, queries, target_l1, seed=0, offset=0):
    """Mix with random feasible visible vertex to match L1 exactly, no oracle IDs.

    A vertex e_k is feasible iff 2(1-a_k)>=d. Uniformly sample eligible k,
    then z=(1-t)a+t*e_k, t=d/[2(1-a_k)]. A feasible target perturbation on
    this same simplex guarantees at least one eligible vertex. No capping.
    """
    a=attention.double();b,h,j,t=a.shape
    q=torch.as_tensor(queries,device=a.device)
    if q.ndim==1:q=q[None].expand(b,-1)
    visible=torch.arange(t,device=a.device)[None,None,None,:]<=q[:,None,:,None]
    # Normalize row-roundoff before constructing the simplex path.
    # Original attention has tiny float normalization error; use exact path
    # distance sum(a)+1-2*a_k rather than assuming it equals 2(1-a_k).
    distance=a.sum(-1,keepdim=True)+1-2*a
    eligible=visible&(distance>=target_l1[...,None]-1e-10)
    if not eligible.any(-1).all():raise ValueError('no feasible random vertex')
    draws=torch.stack([torch.rand((h,j,t),dtype=torch.float64,
        generator=torch.Generator().manual_seed(seed+104729*(offset+i))) for i in range(b)]).to(a.device)
    dest=draws.masked_fill(~eligible,-1).argmax(-1,keepdim=True)
    denom=distance.gather(-1,dest)[...,0]
    weight=torch.where(target_l1>0,target_l1/denom.clamp_min(1e-300),0)
    if (weight>1+1e-7).any():raise ValueError('infeasible control')
    z=a*(1-weight)[...,None];z.scatter_add_(-1,dest,weight[...,None])
    return z.to(attention.dtype)


def outcome(logits, targets):
    ok=logits.argmax(-1).eq(targets)
    c=ok.all(-1);v=ok[...,4]
    first=torch.where(c,c.shape[1]+1,torch.arange(1,c.shape[1]+1,device=c.device)[None]).amin(-1)
    return dict(clause_chain=c.all(-1).cpu(),value_chain=v.all(-1).cpu(),
                clause_step=c.float().mean(-1).cpu(),value_step=v.float().mean(-1).cpu(),
                first_error=first.cpu())


@torch.inference_mode()
def probe(checkpoint, lengths=(16,64), samples=256, batch=32, seed=10140001, shuffled=False):
    torch.set_num_threads(4)
    model,state=load_routing_model(checkpoint,'cuda');model.eval()
    model.requires_grad_(False)
    before=parameter_hash(model.named_parameters())
    task=Task(state['args']['task'],model.config.nvars)
    cs=conditions();allrows=[];saved={}
    for length in lengths:
        sample=task.generate(samples,length,torch.Generator().manual_seed(seed+length))
        if shuffled: sample=permute_predicates(sample,torch.Generator().manual_seed(seed+1000000+length))
        by={c['name']:[] for c in cs};stats={c['name']:[] for c in cs}
        sham_delta=0.;random_error=0.
        for offset in range(0,samples,batch):
            c=sample['clauses'][offset:offset+batch].cuda()
            queries=torch.arange(length,2*length,device=c.device)
            rules=(sample['predicate_positions'][offset:offset+len(c)].cuda() if shuffled else torch.arange(length,device=c.device))
            logits,a=model.forward_queries(c[:,:-1],queries,return_attention=True)
            dists={}
            for spec in cs:
                name=spec['name']
                if spec['kind']=='random':
                    target=dists[spec['reference']]
                    z=matched_random(a,queries,target,seed+length*100+spec['draw']*10000019,offset)
                else:z=intervene(a,rules,queries,spec)
                dg=diagnostics(a,z,rules,queries)
                dists[name]=dg['l1']
                if spec['kind']=='random':
                    err=(dg['l1']-target).abs().max().item();random_error=max(random_error,err)
                    if err>2e-6:raise RuntimeError('random L1 mismatch')
                out=(logits if spec['kind']=='original' else model.forward_queries(c[:,:-1],queries,attention_override=z))
                if spec['kind']=='sham':
                    sham_delta=max(sham_delta,(out-logits).abs().max().item())
                    if not torch.equal(out.argmax(-1),logits.argmax(-1)):raise RuntimeError('sham changed predictions')
                by[name].append(outcome(out,c[:,length+1:]))
                stats[name].append({k:v.float().mean((1,2)).cpu() for k,v in dg.items()})
        outcomes={name:{k:torch.cat([x[k] for x in bs]) for k in bs[0]} for name,bs in by.items()}
        diag={name:{k:torch.cat([x[k] for x in bs]) for k in bs[0]} for name,bs in stats.items()}
        saved[str(length)]={'outcomes':outcomes,'diagnostics':diag,'panel_sha256':tensor_hash(sample['clauses'])}
        for spec in cs:
            name=spec['name'];o=outcomes[name];base=outcomes['original']['clause_chain'];cur=o['clause_chain']
            allrows.append(dict(length=length,condition=name,kind=spec['kind'],samples=samples,
                metrics={k:v.double().mean().item() for k,v in o.items()},
                diagnostics={k:v.double().mean().item() for k,v in diag[name].items()},
                paired=dict(rescued=int((~base&cur).sum()),harmed=int((base&~cur).sum()),
                            delta=cur.double().mean().item()-base.double().mean().item()),
                panel_sha256=tensor_hash(sample['clauses']),sham_max_logit_delta=sham_delta,
                random_max_l1_error=random_error))
    after=parameter_hash(model.named_parameters())
    if before!=after:raise RuntimeError('model mutated')
    return dict(complete=True,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
                parameter_sha256_before=before,parameter_sha256_after=after,
                task=task.name,position_mode=model.config.position_mode,seed=seed,shuffled=shuffled,
                conditions=cs,rows=allrows,interpretation='Oracle-position, true-prefix mechanism diagnostic; not deployable inference or final-answer accuracy'),saved

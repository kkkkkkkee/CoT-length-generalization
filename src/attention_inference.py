"""Inference-time attention transforms using only visible inputs/model weights.

No rule index, expected output, or action table is accepted by transforms.
Grouped routing is a task-specific prior: half mass on predicate clauses,
half on the current answer. Predicate choice remains the model's own.
"""
import torch
from torch import nn

METHODS=[{'name':'original','kind':'original','beta':1.}]+[
    {'name':f'{kind}_{beta:g}','kind':kind,'beta':float(beta)}
    for kind,betas in [('sharpen',(2,4,8)),('grouped',(1,2,4,8))] for beta in betas]


def transform(attention,clauses,positions,kind='original',beta=1.):
    if kind=='original':return attention
    if kind not in ('sharpen','grouped') or beta<=0:raise ValueError('invalid transform')
    b,_,j,t=attention.shape
    q=torch.as_tensor(positions,device=clauses.device)
    if q.ndim==1:q=q[None].expand(b,-1)
    visible=torch.arange(t,device=clauses.device)[None,None,None,:]<=q[:,None,:,None]
    loga=attention.double().clamp_min(1e-300).log()*beta
    loga=loga.masked_fill(~visible,-torch.inf)
    if kind=='sharpen':return loga.softmax(-1).to(attention.dtype)
    # Structural type only: no comparison of source/query variable identities.
    predicate=(clauses[:,:,0]!=0)&(clauses[:,:,1]!=0)&(clauses[:,:,2]!=0)&(clauses[:,:,3]==0)&(clauses[:,:,4]==0)
    allowed=visible&predicate[:,None,None,:]
    has=allowed.any(-1,keepdim=True)
    # Malformed generated prefixes must not cause a NaN or oracle repair.
    safe=allowed.clone();safe.scatter_(-1,q[:,None,:,None],~has)
    rule=loga.masked_fill(~safe,-torch.inf).softmax(-1)
    z=.5*rule
    z.scatter_add_(-1,q[:,None,:,None],torch.full_like(q[:,None,:,None],.5,dtype=z.dtype))
    z=torch.where(has,z,attention.double())
    return z.to(attention.dtype)


class InferenceAttention(nn.Module):
    def __init__(self,model,kind='original',beta=1.):
        super().__init__();self.model=model;self.kind=kind;self.beta=beta
    def forward_queries(self,clauses,positions,return_attention=False,attention_override=None):
        if attention_override is not None:raise ValueError("Oracle attention override is forbidden here")
        logits,a=self.model.forward_queries(clauses,positions,return_attention=True)
        if self.kind!='original':
            a=transform(a,clauses,positions,self.kind,self.beta)
            logits=self.model.forward_queries(clauses,positions,attention_override=a)
        return (logits,a) if return_attention else logits
    def forward(self,clauses):
        return self.forward_queries(clauses,torch.arange(clauses.shape[1],device=clauses.device))
    @torch.no_grad()
    def decode_last(self,clauses,cache=None,temperature=1.):
        if temperature!=1.:raise ValueError('Use configured beta, not external temperature')
        if cache is not None:clauses=torch.cat([cache,clauses],1)
        q=torch.tensor([clauses.shape[1]-1],device=clauses.device)
        return self.forward_queries(clauses,q)[:,0],clauses

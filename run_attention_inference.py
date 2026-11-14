"""Validation-selected, non-oracle inference interventions on fixed checkpoints."""
from pathlib import Path
import json, hashlib, shutil, fcntl, os, sys, time, argparse
import torch
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/attention-inference-2026-11-10'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    out=OUT if not args.smoke_only else OUT.with_name(OUT.name+'-smoke');out.mkdir(parents=True,exist_ok=True)
    lock=(out/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=json.loads((ROOT/'runs/attention-causal-2026-11-10/campaign-plan.json').read_text())['jobs']
    if args.smoke_only:jobs=[jobs[0],jobs[-4]]
    sources=[Path(__file__),*sorted((ROOT/'src').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    manifest=dict(jobs=jobs,source_sha256=hashes,validation_seed=11140001,test_seed=12140001,
                  validation_lengths=[4,8],test_lengths=[4,16,64],validation_samples=8 if args.smoke_only else 128,
                  test_samples=8 if args.smoke_only else 256,rollout_samples=4 if args.smoke_only else 32,
                  normal_and_shuffled=True,selection='One beta per checkpoint family and transform kind, maximize mean validation full-clause TF accuracy over lengths/orderings/model seeds; ties smallest beta. All selection locked before any L64 test.',
                  limits=['Grouped uses clause-type/current-answer architectural prior, not true rule position.',
                          'Temperature power transforms complete attention including positional bias.',
                          'No training; extra inference passes not a matched compute comparison.',
                          '4 model seeds, replicate0; finite evaluation, no general convergence claim.',
                          'TF and greedy full trajectories reported; greedy final-value may recover after errors.'])
    plan=out/'campaign-plan.json'
    if plan.exists():
        if json.loads(plan.read_text())!=manifest:raise RuntimeError('Manifest drift')
    else:
        for p in sources:
            dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        write(plan,manifest)
    for rel,h in hashes.items():
        if sha(out/'source'/rel)!=h:raise RuntimeError('Snapshot drift')
    sys.path.insert(0,str(out/'source/src'))
    from attention_inference import METHODS, InferenceAttention
    from routing_model import load_routing_model
    from routing_experiment import parameter_hash
    from data import Task
    from sparse_experiment import evaluate
    torch.set_num_threads(4)
    completed=[];start=time.time()
    def progress(status,phase=None,current=None,error=None):
        write(out/'campaign-progress.json',dict(status=status,complete=status=='complete',phase=phase,current=current,
              completed=completed,total=2*len(jobs),error=error,pid=os.getpid(),elapsed_seconds=time.time()-start,updated_at=time.time()))
    def load(job):
        if sha(job['checkpoint'])!=job['checkpoint_sha256']:raise RuntimeError('Checkpoint drift')
        model,state=load_routing_model(job['checkpoint'],'cuda');model.eval();model.requires_grad_(False)
        return model,Task(state['args']['task'],model.config.nvars)
    try:
        validations={}
        for job in jobs:
            name=job['name'];progress('running','validation',name);path=out/name/'validation.json'
            if path.exists():doc=json.loads(path.read_text())
            else:
                print('VALIDATE',name,flush=True);model,task=load(job);before=parameter_hash(model.named_parameters());rows=[]
                for m in METHODS:
                    wrapper=InferenceAttention(model,m['kind'],m['beta'])
                    for shuffle in [False,True]:
                        ev=evaluate(wrapper,task,manifest['validation_lengths'],samples=manifest['validation_samples'],
                                    rollout_samples=0,seed=manifest['validation_seed'],batch_size=32,predicate_shuffle=shuffle)
                        rows.append(dict(method=m,shuffled=shuffle,evaluations=ev))
                assert parameter_hash(model.named_parameters())==before
                doc=dict(complete=True,checkpoint_sha256=job['checkpoint_sha256'],rows=rows,model_hash=before)
                write(path,doc);del model
            if not doc['complete'] or doc['checkpoint_sha256']!=job['checkpoint_sha256']:raise RuntimeError('Invalid saved validation')
            validations[name]=doc;completed.append('validation/'+name);progress('running','validation',name)
        # Global phase barrier: selection file is immutable before test evaluation starts.
        families=sorted(set(j['family']+'/'+j['treatment'] for j in jobs));chosen={}
        for family in families:
            members=[j for j in jobs if j['family']+'/'+j['treatment']==family]
            scores={}
            for m in METHODS:
                vals=[e['teacher_forced']['metrics']['clause_chain_accuracy'] for j in members
                      for row in validations[j['name']]['rows'] if row['method']['name']==m['name'] for e in row['evaluations']]
                scores[m['name']]=sum(vals)/len(vals)
            chosen[family]=dict(scores=scores,selected=[METHODS[0]]+[max([m for m in METHODS if m['kind']==kind],key=lambda m:(scores[m['name']],-m['beta'])) for kind in ['sharpen','grouped']])
        selection=out/'selection.json'
        if selection.exists():
            if json.loads(selection.read_text())!=chosen:raise RuntimeError('Selection drift')
        else:write(selection,chosen)
        for job in jobs:
            name=job['name'];progress('running','test',name);path=out/name/'result.json'
            family=job['family']+'/'+job['treatment']
            if path.exists():
                doc=json.loads(path.read_text());assert doc['complete'] and doc['selection_sha256']==sha(selection)
            else:
                print('TEST',name,flush=True);model,task=load(job);before=parameter_hash(model.named_parameters());rows=[]
                for m in chosen[family]['selected']:
                    wrapper=InferenceAttention(model,m['kind'],m['beta'])
                    for shuffle in [False,True]:
                        ev=evaluate(wrapper,task,manifest['test_lengths'],samples=manifest['test_samples'],
                                    rollout_samples=manifest['rollout_samples'],seed=manifest['test_seed'],batch_size=32,predicate_shuffle=shuffle)
                        rows.append(dict(method=m,shuffled=shuffle,evaluations=ev))
                after=parameter_hash(model.named_parameters());assert before==after
                doc=dict(complete=True,checkpoint_sha256=job['checkpoint_sha256'],selection_sha256=sha(selection),
                         model_hash_before=before,model_hash_after=after,family=family,seed=job['seed'],rows=rows)
                write(path,doc);del model
            completed.append('test/'+name);progress('running','test',name)
        progress('complete');write(out/'campaign-complete.json',json.loads((out/'campaign-progress.json').read_text()))
    except Exception as exc:
        progress('failed',error=repr(exc));raise

if __name__=='__main__':main()

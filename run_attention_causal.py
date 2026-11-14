"""Fixed-checkpoint attention concentration/balance diagnostic campaign."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import torch
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/attention-causal-2026-11-10'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def jobs():
    rows=[]
    for treatment in ('dense','soft_1','row_tied'):
        for seed in range(4):
            checkpoint=ROOT/f'runs/phase-d-structural-2026-11-10/main_{treatment}_s{seed}_r0/checkpoint-final.pt'
            rows.append(dict(name=f'phase_d_{treatment}_s{seed}',family='phase_d',treatment=treatment,seed=seed,replicate=0,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
    for mode in ('none','learned_abs','sinusoidal','relative'):
        for seed in range(4):
            checkpoint=ROOT/f'runs/position-pretrained-2026-11-10/routing_{mode}_row_tied_s{seed}_r0/checkpoint-final.pt'
            rows.append(dict(name=f'pretrained_{mode}_row_tied_s{seed}',family='pretrained',treatment=mode+'_row_tied',seed=seed,replicate=0,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    out=OUT if not args.smoke_only else OUT.with_name(OUT.name+'-smoke')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=[Path(__file__).resolve(),*sorted((ROOT/'src').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source}
    selected=jobs()
    if args.smoke_only:selected=[selected[8],selected[-4]]
    samples=16 if args.smoke_only else 256
    manifest=dict(jobs=selected,source_sha256=hashes,samples=samples,batch=16 if args.smoke_only else 32,lengths=[4,16,64],panel_seed=10140001,
                  policy='All conditions evaluated on same new held-out true-prefix traces; no training or seed selection',
                  primary='full-clause exact trajectory accuracy; per-sequence paired outcomes retained',
                  random_control='3 feasible random vertex directions per query, exact L1 match; not matched support size or uniform L1 sphere',
                  limits=['Oracle true rule positions; mechanism diagnostic only','Learned checkpoints fixed; 4 FFN seeds, replicate 0 selected by index','No causal claim about training dynamics; effects conditional on this intervention','Attention total is joint relevant mass, not entropy; two targets must be treated separately'])
    plan=out/'campaign-plan.json'
    if plan.exists():
        if json.loads(plan.read_text())!=manifest:raise RuntimeError('Manifest changed')
    else:
        for p in source:
            dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        write(plan,manifest)
    for rel,h in hashes.items():
        if sha(out/'source'/rel)!=h:raise RuntimeError('Source snapshot changed')
    sys.path.insert(0,str(out/'source/src'))
    probe=importlib.import_module('attention_causal_probe').probe
    complete=[];start=time.time()
    def progress(status,current=None,error=None):
        write(out/'campaign-progress.json',dict(status=status,complete=status=='complete',total=len(selected),completed=complete,current=current,error=error,pid=os.getpid(),elapsed_seconds=time.time()-start,updated_at=time.time()))
    for job in selected:
        name=job['name'];dest=out/name;progress('running',name)
        try:
            if (dest/'result.json').exists():
                existing=json.loads((dest/'result.json').read_text())
                if not existing['complete'] or existing['checkpoint_sha256']!=job['checkpoint_sha256']:raise RuntimeError('Invalid prior result')
                if not (dest/'paired-outcomes.pt').exists():raise RuntimeError('Missing paired outcomes')
            else:
                if sha(job['checkpoint'])!=job['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
                print('START',name,flush=True)
                result,arrays=probe(job['checkpoint'],lengths=manifest['lengths'],samples=samples,batch=manifest['batch'],seed=manifest['panel_seed'])
                dest.mkdir(parents=True,exist_ok=True)
                torch.save(arrays,dest/'paired-outcomes.pt');write(dest/'result.json',result)
                if sha(job['checkpoint'])!=job['checkpoint_sha256']:raise RuntimeError('Checkpoint changed during probe')
            complete.append(name);progress('running',name);print('DONE',name,flush=True)
        except Exception as exc:
            progress('failed',name,repr(exc));raise
    progress('complete');write(out/'campaign-complete.json',json.loads((out/'campaign-progress.json').read_text()))

if __name__=='__main__':main()

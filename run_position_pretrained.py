"""Two-stage position-score experiment with immutable source and resumable jobs."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/position-pretrained-2026-11-10'
MODES = ('none','learned_abs','sinusoidal','relative')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(p)

def specs(out):
    jobs=[]
    # Run each position mode's local and routing jobs together to expose failures early.
    for mode in MODES:
        for seed in range(4):
            name=f'local_{mode}_s{seed}'
            args=['--stage','local','--output',str(out/name),'--seed',str(seed),
                  '--position-mode',mode,'--samples','32768','--pool-samples','32768',
                  '--steps','16000','--batch','256','--lr','.001',
                  '--data-seed',str(9010001+10000*seed),'--eval-seed','9140001',
                  '--eval-every','4000','--local-min-steps','4000',
                  '--local-gate-protocol','position_local','--local-gate','.9995',
                  '--eval-samples','1024','--rollout-samples','0',
                  '--budget-label','position_pretrained_fixed_16000_local']
            jobs.append(dict(name=name,stage='local',mode=mode,seed=seed,parent=None,
                             command=[sys.executable,str(out/'source/src/sparse_experiment.py'),*args]))
            for routing in ('dense','row_tied'):
                for rep in range(2):
                    child=f'routing_{mode}_{routing}_s{seed}_r{rep}'
                    args=['--output',str(out/child),'--init',str(out/name/'checkpoint-final.pt'),
                          '--routing-mode',routing,'--position-mode',mode,'--seed',str(seed),
                          '--replicate',str(rep),'--data-seed',str(9210001+10000*seed+100*rep),
                          '--noise-seed',str(9310001+10000*seed+100*rep),'--eval-seed','9440001',
                          '--validation-seed','9540001','--length','4','--samples','4096',
                          '--pool-samples','16384','--steps','2000','--batch','128','--lr','.03',
                          '--mode','second','--label-source','truth','--eval-every','2000',
                          '--eval-samples','1024','--validation-samples','512','--rollout-samples','128',
                          '--eval-lengths','2','4','8','16','32','64','--eval-predicate-shuffle',
                          '--budget-label','position_pretrained_frozen_ffn_trainable_position']
                    jobs.append(dict(name=child,stage='routing',mode=mode,seed=seed,replicate=rep,
                                     parent=name,command=[sys.executable,str(out/'source/src/routing_experiment.py'),*args]))
    return jobs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=specs(OUT)
    sources=[Path(__file__).resolve(),*sorted((ROOT/'src').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    manifest={'jobs':jobs,'source_sha256':hashes,
              'counts':{'local':16,'routing':64,'total':80},
              'position_mechanism':'score-only positional bias, no additive PE in FFN/value vectors',
              'position_policy':'learned_abs and relative train during local and routing for both dense and row_tied; sinusoidal fixed',
              'local_budget':'L1, 32768 traces, 16000 Adam updates, lr .001, full-clause supervision',
              'routing_budget':'L4, frozen FFN, 4096 traces, 2000 updates, lr .03, second-value supervision',
              'gate':'final independent L1 value/variable and exhaustive action/state >= .9995; long oracle panels diagnostic only',
              'limits':['Two-stage mechanism model, not a full standard Transformer.',
                        'Local PE exposure covers only two visible L1 positions.',
                        'Fixed ordered training; paired normal/shuffled final evaluation.',
                        'Failed local seeds recorded; no replacement or OOD seed selection.',
                        'Previous v2/v3 dense position parameters were frozen, unlike row_tied; not a fair cross-routing PE comparison.']}
    plan=OUT/'campaign-plan.json'
    if plan.exists():
        old=json.loads(plan.read_text())
        if old!=manifest: raise RuntimeError('Manifest/source changed; use a new campaign')
    else:
        for p in sources:
            dest=OUT/'source'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        write(plan,manifest)
    for rel,digest in hashes.items():
        if sha(OUT/'source'/rel)!=digest: raise RuntimeError('Snapshot mismatch')
    if args.dry_run:
        print(json.dumps(manifest['counts']),flush=True);return
    completed=[];blocked=[];failed_gates=[];failures=[]
    started=time.time()
    def progress(status,current=None):
        write(OUT/'campaign-progress.json',dict(status=status,complete=status=='complete',
              total=len(jobs),completed=completed,blocked=blocked,failed_gates=failed_gates,
              failures=failures,current=current,pid=os.getpid(),updated_at=time.time(),elapsed_seconds=time.time()-started))
    write(OUT/'launcher.json',dict(pid=os.getpid(),started_at=time.time()))
    for job in jobs:
        name=job['name'];directory=OUT/name;marker=directory/'complete.json'
        if job['parent']:
            parent=json.loads((OUT/job['parent']/'complete.json').read_text())
            if not parent.get('local_gate_passed'):
                blocked.append(dict(run=name,reason='parent failed independent local gate'))
                progress('running',name);continue
        progress('running',name)
        try:
            if not marker.exists():
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError(f'Incomplete run preserved: {directory}; inspect before restarting')
                if shutil.disk_usage(OUT).free<5*2**30: raise RuntimeError('Less than 5 GiB free')
                print('START',name,flush=True)
                log=OUT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
                with log.open('w') as handle:
                    subprocess.run(job['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
            result=json.loads(marker.read_text())
            if not result.get('complete'): raise RuntimeError('Invalid completion')
            if result['frozen_sha256_before']!=result['frozen_sha256_after']: raise RuntimeError('Frozen state changed')
            if job['stage']=='local' and not result['local_gate_passed']:
                failed_gates.append(name)
            if job['stage']=='routing':
                if result['ff_sha256_before']!=result['ff_sha256_after']: raise RuntimeError('FFN changed')
                ev=json.loads((directory/'evaluation.json').read_text())
                if len(ev.get('shuffled_evaluations') or [])!=6: raise RuntimeError('Missing shuffled evaluation')
            completed.append(name)
            print('DONE',name,flush=True);progress('running',name)
        except Exception as exc:
            failures.append(dict(run=name,error=repr(exc)))
            progress('failed',name);raise
    progress('complete' if not blocked else 'finished_with_gate_failures')
    write(OUT/'campaign-complete.json',json.loads((OUT/'campaign-progress.json').read_text()))

if __name__=='__main__': main()

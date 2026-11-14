"""Run the isolated position-bias routing experiment."""
from pathlib import Path
import argparse, json, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/position-embedding-v3-2026-11-10'
MODES=('none','learned_abs','sinusoidal','relative')
ROUTINGS=('dense','row_tied')
ROUTES=('isolated','standard')

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 runs=[{'route':'isolated','position_mode':m,'routing':q,'seed':s,'replicate':rep}
       for m in MODES for q in ROUTINGS for s in range(4) for rep in range(2)]
 (OUT/'manifest.json').write_text(json.dumps({'runs':runs,'normal_eval':True,'permuted_predicate_eval':False,'implemented_route':'isolated'},indent=2)+'\n')
 if a.dry_run: print(json.dumps({'output':str(OUT),'runs':len(runs),'dry_run':True})); return
 completed=[]; failures=[]
 for x in runs:
  name=f"isolated_{x['position_mode']}_{x['routing']}_s{x['seed']}_r{x['replicate']}"; out=OUT/name
  if (out/'complete.json').exists(): completed.append(name); continue
  base=ROOT/('runs/sparse-supervision-2026-11-09' if x['seed']<3 else 'runs/large-scale-2026-11-10/local')
  init=base/('local_full_s%d/checkpoint-final.pt'%x['seed'])
  cmd=[sys.executable,'src/routing_experiment.py','--output',str(out),'--init',str(init),'--routing-mode',x['routing'],'--position-mode',x['position_mode'],'--seed',str(x['seed']),'--replicate',str(x['replicate']),'--data-seed',str(8710001+x['seed']*10000+x['replicate']*100),'--noise-seed',str(8810001+x['seed']*10000+x['replicate']*100),'--length','4','--samples','4096','--pool-samples','16384','--steps','2000','--batch','128','--lr','.03','--mode','second','--label-source','truth','--eval-lengths','2','4','8','16','32','64','--eval-every','2000']
  out.mkdir(parents=True,exist_ok=True)
  with (OUT/(name+'.log')).open('w') as h: subprocess.run(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,check=True)
  completed.append(name); (OUT/'progress.json').write_text(json.dumps({'completed':completed,'total':len(runs)},indent=2))
 print(json.dumps({'output':str(OUT),'completed':len(completed),'total':len(runs)}))
if __name__=='__main__': main()

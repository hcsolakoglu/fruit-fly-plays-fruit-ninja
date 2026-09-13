"""Matched-seed evaluation and recordings. No evaluation-driven model fitting."""
import sys,json,time,argparse,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flyfruit.world import World
from flyfruit.brain import Brain

def run_one(args):
 condition,seed,hard,frames,record=args;world=World(seed,hard);brain=Brain(condition);states=[];activity=[];fields=[];retina=[];events=[]
 nodes=json.loads((ROOT/'data/nodes.json').read_text());v=np.load(ROOT/'data/display.npz');vis=v['indices']
 for f in range(frames):
  action,trace=brain.act(world.pixels(),world.blade);ev=world.step(action)
  if record:
   snap=world.snapshot();snap['action']=list(action);states.append(snap);activity.append(trace['activity'][vis]);fields.append(trace['field']);retina.append(trace['retina']);events.extend(ev)
 m={'condition':condition,**world.metrics()}
 if record:
  name=f'{condition}_{seed}';folder=ROOT/'results/recordings';folder.mkdir(exist_ok=True,parents=True)
  (folder/(name+'.json')).write_text(json.dumps({'meta':m,'states':states,'events':events},separators=(',',':')))
  np.savez_compressed(folder/(name+'.npz'),activity=np.array(activity,dtype=np.float16),field=np.array(fields,dtype=np.float16),retina=np.array(retina,dtype=np.float16))
 return m

def display_graph():
 from scipy import sparse
 nodes=json.loads((ROOT/'data/nodes.json').read_text());w=sparse.load_npz(ROOT/'data/connectome.npz');strength=np.asarray(abs(w).sum(0)).ravel();indices=[]
 for cl,n in [('visual_projection',220),('cb_intrinsic',180),('descending_neuron',160)]:
  ix=np.array([i for i,v in enumerate(nodes) if v['class']==cl and v['soma'] is not None]);indices.extend(ix[np.argsort(strength[ix])[-n:]].tolist())
 indices=np.array(sorted(indices));lookup={int(v):i for i,v in enumerate(indices)};e=w.tocoo();eligible=np.array([k for k,(a,b) in enumerate(zip(e.col,e.row)) if int(a) in lookup and int(b) in lookup]);choose=eligible[np.argsort(abs(e.data[eligible]))[-1800:]]
 pre=np.array([lookup[int(x)] for x in e.col[choose]]);post=np.array([lookup[int(x)] for x in e.row[choose]])
 coords=np.array([nodes[i]['soma'] for i in indices],dtype=np.float32);coords=(coords-np.median(coords,axis=0))/np.ptp(coords,axis=0)
 np.savez_compressed(ROOT/'data/display.npz',indices=indices,coords=coords,pre=pre,post=post)
 (ROOT/'data/display_nodes.json').write_text(json.dumps([nodes[i] for i in indices],separators=(',',':')))

def main():
 p=argparse.ArgumentParser();p.add_argument('--dev',action='store_true');p.add_argument('--frames',type=int,default=600);p.add_argument('--workers',type=int,default=6);a=p.parse_args();display_graph();start=time.time()
 cond=['intact','rewired','disconnected','blind','random','pixel'];seeds=range(0,3) if a.dev else range(100,120)
 jobs=[(c,s,False,a.frames,a.dev and s==1) for c in cond for s in seeds]
 if not a.dev:jobs.extend([('intact',s,True,a.frames,False) for s in range(200,210)])
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
  for r in pool.map(run_one,jobs):rows.append(r);print(json.dumps(r),flush=True)
 name='development' if a.dev else 'evaluation';(ROOT/f'results/{name}.json').write_text(json.dumps(rows,indent=2))
 summary={}
 for c in cond+['intact_hard']:
  rr=[r for r in rows if (r['condition']+('_hard' if r['hard'] else ''))==c]
  if not rr:continue
  scores=np.array([r['reward'] for r in rr]);summary[c]={'episodes':len(rr),'fruit_sliced':sum(r['fruit_sliced'] for r in rr),'fruit_spawned':sum(r['fruit_spawned'] for r in rr),'bombs_hit':sum(r['bombs_hit'] for r in rr),'bombs_spawned':sum(r['bombs_spawned'] for r in rr),'mean_reward':float(scores.mean()),'reward_sd':float(scores.std(ddof=1)) if len(rr)>1 else 0.,'mean_hit_rate':float(np.mean([r['hit_rate'] for r in rr]))}
 (ROOT/f'results/{name}_summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),'SECONDS',time.time()-start,flush=True)
if __name__=='__main__':main()

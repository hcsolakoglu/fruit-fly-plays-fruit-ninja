"""Colab GPU: batched connectome propagation and ridge motor-field decoding."""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
root=Path('/content/fruit-fly') if Path('/content/fruit-fly').exists() else Path(__file__).resolve().parents[1]
os.chdir(root);sys.path.insert(0,str(root/'src'))
from flyfruit.vision import sample_scene

def run():
 assert torch.cuda.is_available(),'Use Colab GPU for training; no local GPU fallback.'
 torch.set_num_threads(4);torch.manual_seed(42);device='cuda';start=time.time();Path('results').mkdir(exist_ok=True)
 rng=np.random.default_rng(42);ntrain=6500;nval=600
 X=np.stack([sample_scene(rng) for _ in range(ntrain+nval)])
 w=sparse.load_npz('data/connectome.npz');it=np.load('data/interface.npz');inp=it['input_idx'];out=it['output_idx'];bins=torch.tensor(it['sensor_bins'],device=device);iw=torch.tensor(it['sensor_weights'],device=device,dtype=torch.float32)
 # Target permutation keeps every source's outdegree and edge-weight multiset;
 # postsynaptic identities, indegrees, and anatomical pathways are not preserved.
 co=w.tocoo();rr=np.random.default_rng(314159);post=rr.permutation(co.row)
 wr=sparse.coo_matrix((co.data,(post,co.col)),shape=w.shape).tocsr();wr=sparse.diags(1/np.maximum(np.asarray(abs(wr).sum(1)).ravel(),1e-8))@wr;wr=wr.tocsr().astype(np.float32);sparse.save_npz('data/rewired.npz',wr)
 report={}
 for name,mat in [('intact',w),('rewired',wr)]:
  mt=mat.tocoo();idx=torch.tensor(np.stack([mt.row,mt.col]),device=device);wt=torch.sparse_coo_tensor(idx,torch.tensor(mt.data,device=device),mt.shape).coalesce();outputs=[]
  for base in range(0,len(X),256):
   xb=torch.tensor(X[base:base+256],device=device);drive=torch.zeros((mat.shape[0],len(xb)),device=device);drive[inp]=(xb[:,bins]*iw).sum(-1).T*3;h=torch.zeros_like(drive)
   for _ in range(6):h=torch.tanh(1.6*torch.sparse.mm(wt,h)+drive)
   outputs.append(h[out].T)
  z=torch.cat(outputs);mean=z[:ntrain].mean(0);std=z[:ntrain].std(0).clamp_min(1e-5);z=(z-mean)/std;z=torch.cat([z,torch.ones((len(z),1),device=device)],1)
  y=torch.tensor(X,device=device);eye=torch.eye(z.shape[1],device=device);eye[-1,-1]=0
  # Fixed lambda; development validation is reported, not used for repeated tuning.
  coef=torch.linalg.solve(z[:ntrain].T@z[:ntrain]+10*eye,z[:ntrain].T@y[:ntrain]);pred=z[ntrain:]@coef
  report[name]={'train_scenes':ntrain,'validation_scenes':nval,'mse':float(((pred-y[ntrain:])**2).mean()),'zero_predictor_mse':float((y[ntrain:]**2).mean()),'active_output_neurons':int((std>1e-5).sum()),'ridge_lambda':10.0}
  filename='readout.npz' if name=='intact' else 'readout_rewired.npz';np.savez_compressed('data/'+filename,coef=coef.cpu().numpy(),mean=mean.cpu().numpy(),std=std.cpu().numpy());print(name,report[name],flush=True)
 report['hardware']=torch.cuda.get_device_name(0);report['torch']=torch.__version__;report['seconds']=time.time()-start;report['seed']=42;report['learning_claim']='Supervised engineered readout fit only; fixed biological edges; no synaptic plasticity or reinforcement learning.'
 Path('results/training.json').write_text(json.dumps(report,indent=2));print('DONE',report['seconds'],flush=True)
if __name__=='__main__':run()

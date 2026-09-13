"""Extract an anatomically selected, neuron-level MaleCNS circuit."""
import os,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

def build(root='.'):
 root=Path(root);d=root/'data'
 a=pd.read_feather(d/'annotations.feather').set_index('bodyId',drop=False)
 e=pd.read_feather(d/'weights.feather');nt=pd.read_feather(d/'nt.feather')
 print('EDGE SCHEMA',e.columns.tolist(),'NT SCHEMA',nt.columns.tolist(),flush=True)
 # Source schema is checked explicitly, never guessed silently.
 assert {'body_pre','body_post','weight'} <= set(e.columns)
 visual=a.index[a.superclass.eq('visual_projection')].to_numpy()
 dn=a.index[a.superclass.eq('descending_neuron')].to_numpy()
 cb=a.index[a.superclass.eq('cb_intrinsic')].to_numpy()
 vin=e[e.body_pre.isin(visual)].groupby('body_post').weight.sum()
 dout=e[e.body_post.isin(dn)].groupby('body_pre').weight.sum()
 bridge=(vin.reindex(cb,fill_value=0)*dout.reindex(cb,fill_value=0)).sort_values(ascending=False)
 inter=bridge[bridge>0].head(1024).index.to_numpy()
 vscore=e[e.body_pre.isin(visual)&e.body_post.isin(np.r_[inter,dn])].groupby('body_pre').weight.sum()
 vi=vscore.sort_values(ascending=False).head(2048).index.to_numpy()
 ids=np.sort(np.unique(np.r_[vi,inter,dn]));ann=a.loc[ids].copy();lookup=pd.Series(np.arange(len(ids)),index=ids)
 edges=e[e.body_pre.isin(ids)&e.body_post.isin(ids)].copy()
 pre=lookup.loc[edges.body_pre].to_numpy();post=lookup.loc[edges.body_post].to_numpy();count=edges.weight.to_numpy(dtype=np.float32)
 # Exact source field selection based on inspected schema.
 idcol='body' if 'body' in nt else 'bodyId'
 predcol='predicted_nt' if 'predicted_nt' in nt else 'nt_type'
 assert idcol in nt and predcol in nt,(nt.columns.tolist(),idcol,predcol)
 nts=nt.set_index(idcol)[predcol].reindex(ids).fillna('unknown').astype(str).str.lower().to_numpy()
 signs=np.where(np.isin(nts,['gaba','glutamate','glut','glu']),-1.,1.).astype(np.float32)
 raw=sparse.csr_matrix((count,(post,pre)),shape=(len(ids),len(ids)))
 # Per-postsynaptic total-count normalization preserves relative incoming strengths.
 scale=np.asarray(raw.sum(1)).ravel();w=sparse.diags(1/np.maximum(scale,1))@raw@sparse.diags(signs)
 w=w.tocsr().astype(np.float32);sparse.save_npz(d/'connectome.npz',w)
 np.savez_compressed(d/'edges.npz',pre=pre.astype(np.int32),post=post.astype(np.int32),count=count)
 inp=np.flatnonzero(ann.superclass.eq('visual_projection').to_numpy());out=np.flatnonzero(ann.superclass.eq('descending_neuron').to_numpy())
 rng=np.random.default_rng(20260913);sensor_bins=rng.integers(0,192,(len(inp),12));sensor_weights=rng.choice([-1.,1.],sensor_bins.shape).astype(np.float32)/np.sqrt(12)
 # Two foreground channels x 12 columns x 8 rows. Assignment is engineered, not retinotopy.
 np.savez_compressed(d/'interface.npz',input_idx=inp,output_idx=out,sensor_bins=sensor_bins,sensor_weights=sensor_weights)
 nodes=[]
 for i,(_,row) in enumerate(ann.iterrows()):
  loc=row.somaLocation
  nodes.append({'i':i,'body_id':int(row.bodyId),'type':str(row['type']),'class':str(row.superclass),'side':str(row.somaSide),'nt':str(nts[i]),'soma':list(map(float,loc)) if isinstance(loc,(list,np.ndarray)) and len(loc)==3 else None})
 (d/'nodes.json').write_text(json.dumps(nodes,separators=(',',':')))
 meta={'dataset':'MaleCNS v1.0, minconf 0.5','source':'https://male-cns.janelia.org/download/','license':'CC-BY-4.0','nodes':len(ids),'edges':len(edges),'synapses':int(count.sum()),'visual_neurons':len(inp),'descending_neurons':len(out),'interneurons':len(inter),'selection':'Top 1024 central-brain bridges by visual-incoming x descending-outgoing synapse counts; top 2048 visual projection neurons by input to bridge/DN set; all annotated descending neurons. Induced subgraph, no invented biological edges.','unknown_nt':int(sum(nts=='unknown')),'nt_counts':pd.Series(nts).value_counts().to_dict(),'dynamics':'Six tanh rate-propagation steps per observation, zero initial activity; W[post,pre]=NT-sign(pre)*count/sum_incoming(post); uniform gain 1.6. No biophysical time calibration.'}
 (d/'graph_summary.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':build('/content/fruit-fly' if Path('/content/fruit-fly').exists() else '.')

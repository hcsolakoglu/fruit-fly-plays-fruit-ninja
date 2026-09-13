import os,time,json,hashlib,concurrent.futures
from pathlib import Path
import requests,pandas as pd,torch
BASE='https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
os.chdir('/content');Path('fruit-fly/data').mkdir(parents=True,exist_ok=True);os.chdir('/content/fruit-fly')
files={'annotations':'body-annotations-male-cns-v1.0-minconf-0.5.feather','weights':'connectome-weights-male-cns-v1.0-minconf-0.5.feather','nt':'body-neurotransmitters-male-cns-v1.0.feather'}
def get(kv):
 k,f=kv;p=Path('data')/(k+'.feather');t=time.time()
 if not p.exists():
  with requests.get(BASE+f,stream=True,timeout=180) as r:
   r.raise_for_status()
   with p.open('wb') as o:
    for c in r.iter_content(8*1024*1024):o.write(c)
 print(k,p.stat().st_size,round(time.time()-t,1),flush=True)
 return k,{'url':BASE+f,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(3) as pool: provenance=dict(pool.map(get,files.items()))
Path('data/source_manifest.json').write_text(json.dumps(provenance,indent=2))
for k in files:
 d=pd.read_feather('data/'+k+'.feather');print(k,d.shape,d.columns.tolist(),d.head(2).to_dict('records'),flush=True)
print('HARDWARE',torch.cuda.get_device_name(0),torch.__version__,flush=True)

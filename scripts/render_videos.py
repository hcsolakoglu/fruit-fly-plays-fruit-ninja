"""Five edits, original procedural audio, and a frame-level source manifest."""
import os,sys,json,math,subprocess,wave,argparse,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flyfruit.render import Painter,Recording
SPECS=[('01_tiny_sword','hero',20,'intact_1'),('02_pixels_to_neurons','brain',22,'intact_1'),('03_cut_the_wiring','cut',20,'intact_1'),('04_fruit_vs_bomb','hazard',20,'intact_1'),('05_real_vs_rewired','rewire',24,'rewired_1')]

def audio(path,rec,frames,style):
 sr=48000;dur=len(frames)/30;n=int(sr*dur);sound=np.zeros(n,dtype=np.float32);rng=np.random.default_rng(730+len(style));tempo={'hero':112,'brain':88,'cut':102,'hazard':122,'rewire':96}[style]
 def add(t,arr,gain=1):
  start=int(t*sr);end=min(n,start+len(arr))
  if start>=0 and end>start:sound[start:end]+=arr[:end-start]*gain
 # Original synthesized score: pulse bass, restrained chords, percussive texture.
 for beat,t in enumerate(np.arange(0,dur,60/tempo)):
  tt=np.arange(int(.32*sr))/sr;kick=np.sin(2*np.pi*(47*tt+5*(1-np.exp(-tt*28))))*np.exp(-tt*16);add(t,kick,.13)
  freq=[110,130.813,146.832,98][(beat//4)%4];tt=np.arange(int(.48*sr))/sr;note=(np.sin(2*np.pi*freq*tt)+.25*np.sin(2*np.pi*freq*2*tt))*np.exp(-tt*7);add(t,note,.07)
  tt=np.arange(int(.055*sr))/sr;hat=rng.normal(0,1,len(tt))*np.exp(-tt*65);add(t+30/tempo,hat,.018)
  if beat%4==0:
   tt=np.arange(int(1.5*sr))/sr;chord=sum(np.sin(2*np.pi*freq*r*tt) for r in [2,2.5,3])/3;env=np.minimum(tt*5,1)*np.exp(-tt*1.8);add(t,chord*env,.045)
 # Event Foley aligned to the first frame where each recorded event is shown.
 emitted=set()
 for k,f in enumerate(frames):
  for j,e in enumerate(rec.events):
   if j not in emitted and (k==0 or frames[k-1]<e['frame']) and f>=e['frame']:
    emitted.add(j);tt=np.arange(int(.32*sr))/sr
    if e['event']=='slice':s=rng.normal(0,1,len(tt))*np.exp(-tt*27)+.35*np.sin(2*np.pi*(700*tt-500*tt*tt))*np.exp(-tt*15);add(k/30,s,.13)
    else:s=(rng.normal(0,1,len(tt))*.5+np.sin(2*np.pi*57*tt))*np.exp(-tt*10);add(k/30,s,.24)
 sound*=np.minimum(np.arange(n)/sr/.3,1)*np.minimum((n-np.arange(n))/sr/.8,1);peak=max(abs(sound).max(),1e-6);sound*=.88/max(peak,.88)
 with wave.open(str(path),'wb') as w:w.setnchannels(2);w.setsampwidth(2);w.setframerate(sr);w.writeframes(np.repeat((sound*32767).astype('<i2')[:,None],2,axis=1).tobytes())

def render(spec):
 name,style,dur,source=spec;rec=Recording(source);p=Painter();other=Recording('disconnected_1') if style=='cut' else None
 summary=json.loads((ROOT/'results/evaluation_summary.json').read_text()) if (ROOT/'results/evaluation_summary.json').exists() else None
 frames=[]
 # Hazard edit slows an actual recorded bomb collision, then resumes the same trace.
 bombs=[e for e in rec.events if e['event']=='bomb'];focus=bombs[0]['frame'] if bombs else 300
 for k in range(dur*30):
  if style=='hazard':
   if k<90:f=k
   elif k<420:f=focus-100+(k-90)*.55
   else:f=focus+82+(k-420)
  else:f=k
  frames.append(int(np.clip(f,0,len(rec.states)-1)))
 w,h=(1920,1080) if style=='cut' else (1080,1920);out=ROOT/'videos'/f'{name}.mp4';wav=ROOT/'videos'/f'{name}.wav';audio(wav,rec,frames,style)
 cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','30','-i','-','-i',str(wav),'-c:v','libx264','-preset','fast','-crf','19','-threads','2','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',str(out)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 for k,f in enumerate(frames):
  im=p.comparison(rec,other,f,k/30,dur) if style=='cut' else p.portrait(rec,f,style,k/30,dur,summary)
  if k==min(180,len(frames)-1):im.save(ROOT/'videos'/f'{name}.jpg',quality=95)
  proc.stdin.write(im.tobytes())
 proc.stdin.close();code=proc.wait();assert code==0,(name,code);wav.unlink()
 manifest={'video':str(out.relative_to(ROOT)),'source':source,'secondary_source':other.name if other else None,'fps':30,'duration':dur,'source_frame_per_output_frame':frames,'editing':'Hazard: slow motion and editorial jump to first bomb collision. Other edits: real-time source then end freeze if duration exceeds recording.','audio':'Original procedural synthesis; collision Foley follows recorded events.','neural_visual':'Recorded signed rates, brightness from absolute activity; actual soma positions, selected real edges; slight camera rotation is editorial.'}
 (ROOT/'videos'/f'{name}.sources.json').write_text(json.dumps(manifest,indent=2));print('RENDERED',name,flush=True)
 return str(out)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--only',type=int);a.add_argument('--workers',type=int,default=3);a.add_argument('--preview',action='store_true');args=a.parse_args()
 if args.preview:
  p=Painter();r=Recording('intact_1');p.portrait(r,180).save(ROOT/'results/preview.png')
 else:
  specs=[SPECS[args.only-1]] if args.only else SPECS
  with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:list(pool.map(render,specs))

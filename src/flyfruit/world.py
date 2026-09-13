"""Deterministic environment. This module owns hidden state and scoring."""
import numpy as np
from PIL import Image,ImageDraw
W,H=640,480
FPS=30
FRUIT_COLORS=[(250,113,57),(181,93,249),(62,225,176),(251,196,66)]
class World:
 def __init__(self,seed=0,hard=False):
  self.seed=seed;self.rng=np.random.default_rng(seed);self.hard=hard;self.frame=0;self.objects=[];self.events=[];self.blade=np.array([320.,430.]);self.slicing=False;self.sliced=0;self.bombs=0;self.missed=0;self.spawned=0;self.bombs_spawned=0;self.next_id=0
 def spawn(self):
  # Schedule independent of action: paired conditions see identical launches.
  for k in range(int(self.rng.integers(1,4))):
   bomb=bool(self.rng.random()<.25);x=float(self.rng.uniform(90,550));vx=float(self.rng.uniform(-70,70)*(1.6 if self.hard else 1))
   o={'id':self.next_id,'x':x,'y':510.+k*15,'vx':vx,'vy':float(self.rng.uniform(-480,-390)*(1.18 if self.hard else 1)),'r':float(self.rng.uniform(22,29)),'bomb':bomb,'kind':int(self.rng.integers(4)),'angle':float(self.rng.uniform(0,6.28)),'spin':float(self.rng.uniform(-2,2))}
   self.next_id+=1;self.objects.append(o);self.bombs_spawned+=int(bomb);self.spawned+=int(not bomb)
 def pixels(self):
  # Observation is the same geometry/colors as display, without HUD, trail or particles.
  im=Image.new('RGB',(W,H),(7,11,24));d=ImageDraw.Draw(im)
  for o in self.objects:
   x,y,r=o['x'],o['y'],o['r'];c=(174,181,197) if o['bomb'] else FRUIT_COLORS[o['kind']]
   d.ellipse((x-r,y-r,x+r,y+r),fill=c)
  return np.asarray(im)
 def step(self,action):
  old=self.blade.copy();target=np.asarray(action[:2],dtype=float);dist=np.linalg.norm(target-old);self.blade=old+(target-old)*min(1.,(1500/FPS)/max(dist,1e-8));self.blade=np.clip(self.blade,[0,0],[W,H]);self.slicing=bool(action[2]);new_events=[]
  if self.frame%28==0:self.spawn()
  keep=[]
  for o in self.objects:
   o['x']+=o['vx']/FPS;o['y']+=o['vy']/FPS;o['vy']+=430/FPS;o['angle']+=o['spin']/FPS
   p=np.array([o['x'],o['y']]);v=self.blade-old;u=np.clip(np.dot(p-old,v)/max(np.dot(v,v),1e-9),0,1);hit=self.slicing and np.linalg.norm(p-(old+u*v))<o['r']+4 and 0<o['y']<H
   if hit:
    kind='bomb' if o['bomb'] else 'slice';self.bombs+=int(o['bomb']);self.sliced+=int(not o['bomb']);new_events.append({'frame':self.frame,'event':kind,**o});continue
   if o['y']>560 and o['vy']>0:
    self.missed+=int(not o['bomb']);continue
   keep.append(o)
  self.objects=keep;self.events.extend(new_events);self.frame+=1
  return new_events
 def snapshot(self):
  return {'frame':self.frame,'objects':[o.copy() for o in self.objects],'blade':self.blade.tolist(),'slicing':self.slicing,'sliced':self.sliced,'bombs':self.bombs,'spawned':self.spawned,'missed':self.missed}
 def metrics(self):
  return {'seed':self.seed,'hard':self.hard,'frames':self.frame,'fruit_spawned':self.spawned,'fruit_sliced':self.sliced,'bombs_spawned':self.bombs_spawned,'bombs_hit':self.bombs,'fruit_missed':self.missed,'reward':self.sliced-3*self.bombs,'hit_rate':self.sliced/max(1,self.spawned)}

from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base=Path('/root/autodl-tmp/tf-logs')
root=base/'fracture2d/pinn_core+halo_tip_tol2e5'
full=base/'fracture2d/pinn_baseline'
reference=base/'fracture2d/local_damage_load_refined_20261011_direct'
a=json.loads((root/'result.json').read_text())
b=json.loads((full/'result.json').read_text())
cur={r['index']:r for r in a['curve']}
old={r['index']:r for r in b['curve']}
ids=sorted(set(cur)&set(old))
events={}
for name,directory in [('full',base/'pinn_baseline'),('random',base/'pinn_core+halo_tip_tol2e5')]:
 e=EventAccumulator(str(directory),size_guidance={'scalars':0})
 e.Reload()
 events[name]=e
wall={name:{v.step:v.wall_time for v in e.Scalars('train/reaction')} for name,e in events.items()}
rows=[]
for i in ids:
 assert cur[i]['load']==old[i]['load']
 rows.append(dict(index=i,load=cur[i]['load'],full_outer=old[i]['sweep'],random_outer=cur[i]['sweep'],full_status=old[i]['status'],random_status=cur[i]['status'],full_dchange=old[i]['damage_rms'],random_dchange=cur[i]['damage_rms'],full_seconds=None if i==1 else wall['full'][i]-wall['full'][i-1],random_seconds=None if i==1 else wall['random'][i]-wall['random'][i-1]))
matched=rows[1:]
summary={}
for name in ('full','random'):
 summary[name]=dict(seconds=sum(r[name+'_seconds'] for r in matched),outers=sum(r[name+'_outer'] for r in matched),budget_exits=sum(r[name+'_status']=='budget_exhausted' for r in matched),matched_positive_loads=len(matched))
summary['wall_ratio']=summary['random']['seconds']/summary['full']['seconds']
summary['outer_ratio']=summary['random']['outers']/summary['full']['outers']
start=23*1024
stages={}
for name,e in events.items():
 times={v.step:v.wall_time for v in e.Scalars('diag/damage_rms')}
 stages[name]=dict(load=.0024,from_outer=16,to_outer=96,seconds_per_outer=(times[start+96]-times[start+16])/80)
with np.load('/root/autodl-tmp/data/fracture2d/scale_separation_20261001/cloud_h004_api.npz',allow_pickle=False) as cloud:
 v=cloud['volumes'];x=cloud['coordinates']
tip=np.linalg.norm(x-np.array([.5,.5]),axis=1)<=.024
errors=[]
for i in sorted(set([10,15,20,max(ids)])&set(ids)):
 with np.load(reference/f'state_{i:04d}/fields.npz',allow_pickle=False) as f:
  dref=f['d']
 refrow=json.loads((reference/f'state_{i:04d}/result.json').read_text())
 error=dict(index=i,load=cur[i]['load'])
 for name,path,row in [('full',full,old[i]),('random',root,cur[i])]:
  with np.load(path/f'state_{i:04d}/fields.npz',allow_pickle=False) as f:
   d=f['d']
  error[name]=dict(global_L2=float(np.sqrt(v@(d-dref)**2/(v@dref**2))),tip_L2=float(np.sqrt(v[tip]@(d[tip]-dref[tip])**2/(v[tip]@dref[tip]**2))),reaction_relative_error=abs(row['reaction']/refrow['reaction']-1))
 errors.append(error)
report=dict(observed=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),completed_through=cur[max(ids)]['load'],matched_interval=[rows[1]['load'],rows[-1]['load']],summary=summary,same_load_stage_timing=stages,same_load_field_errors=errors,rows=rows,current_at_capture=a['current'],notes=['Different sampling and damage-change tolerance; trajectories have different warm histories.','Endpoint wall-time differences exclude first-load startup and include routine output/plot/checkpoint costs.','No experiment or stopping-condition change; existing worker kept running.'])
out=root/'performance_review_20261015'
out.mkdir(exist_ok=True)
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:val for k,val in report.items() if k not in ('rows','current_at_capture')},indent=2))
print('Recent rows:',json.dumps(rows[-8:],indent=2))

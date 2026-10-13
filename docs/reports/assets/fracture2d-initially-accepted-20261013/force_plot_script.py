"""Read-only CPU force/displacement comparison; no training or FEM launches."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

BASE = Path('/root/autodl-tmp/tf-logs/fracture2d')
NEW = BASE / 'l1_damage_loading_from_zero_20261013'
OLD = BASE / 'bounded_damage_loading_20261013'
FEM = BASE / 'local_gradient_from_zero_20261011'
REF = BASE / 'local_damage_load_refined_20261011_direct'
PREFIX = BASE / 'linear_damage_loading_20261012'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = NEW / 'independent_review' / ('force_displacement_' + STAMP)

def read(p):
    return json.loads(p.read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok, message):
    if not ok:
        raise ValueError(message)

identities = {p.name: read(p/'identity.json') for p in [NEW, OLD, FEM, REF, PREFIX]}
for name, identity in identities.items():
    require(identity['geometry'] == identities[NEW.name]['geometry'], 'Geometry mismatch: '+name)
    require(identity['config']['physical'] == identities[NEW.name]['config']['physical'], 'Material mismatch: '+name)
require(identities[FEM.name]['damage_operator'] == 'material_P1_gradient_stiffness_over_lumped_volume', 'Wrong local damage model')
require(identities[OLD.name]['config']['core']['power'] == 16, 'Wrong old loss')
newresult = read(NEW/'result.json')
audit_path = NEW/'independent_review/audit.json'
audit = read(audit_path)
require(audit['pass_replay'], 'Independent L1 audit failed')
completed = {r['index']:r for r in audit['rows'] if r['final_load'] and r['pass_replay']}
manifest = []
data = {}

def record(label, path, r, field_hash_key, status):
    require(np.isfinite(r['reaction']) and np.isfinite(r['load']), 'Nonfinite curve point')
    manifest.append({'series':label,'report':str(path),'report_sha256':sha(path), 'load':r['load'], 'reaction':r['reaction'], 'fields_sha256':r[field_hash_key], 'status':status})
    return {'load':float(r['load']), 'reaction':float(r['reaction']), 'status':status}

for label, root in [('FEM',FEM),('Direct',REF)]:
    rows = []
    for r0 in read(root/'result.json')['curve']:
        if r0['load'] > .0060+1e-14:
            continue
        p = root/f"state_{r0['index']:04d}"/'result.json'
        r = read(p)
        require(r['status']=='ready' and abs(r['load']-r0['load'])<1e-14 and abs(r['reaction']-r0['reaction'])<1e-8, 'Direct curve/state mismatch')
        rows.append(record(label,p,r,'state_sha256','ready'))
    data[label] = [{'load':0.,'reaction':0.,'status':'exact_zero'}]+rows

rows = []
for r0 in newresult['curve']:
    if r0['index'] not in completed:
        continue
    p = NEW/f"state_{r0['index']:04d}"/'result.json'
    r = read(p)
    ar = completed[r['index']]
    require(abs(r['load']-ar['load'])<1e-14 and abs(r['reaction']-ar['reaction'])<1e-7, 'L1 audit reaction mismatch')
    require(sha(p.parent/'fields.npz')==r['fields_sha256'], 'L1 immutable fields changed')
    rows.append(record('L1 Adam',p,r,'fields_sha256',r['status']))
data['L1 Adam'] = [{'load':0.,'reaction':0.,'status':'exact_zero'}]+rows

old_manifest = read(OLD/'comparison_current_sigmoid_0058_20261013/comparison_report.json')['curve_sources']
old_rows = []
for entry in old_manifest:
    if entry['label'] != 'PINN' or entry['index'] < 52 or entry['index'] > 57:
        continue
    p = Path(entry['report'])
    require(sha(p)==entry['report_sha256'], 'Old comparison source changed')
    r = read(p)
    require(sha(p.parent/'fields.npz')==entry['fields_sha256'], 'Old fields changed')
    old_rows.append(record('L16 LBFGS',p,r,'state_sha256',r['status']))
data['L16 LBFGS'] = old_rows
old_entry = next(e for e in old_manifest if e['label']=='PINN' and e['index']==58)
p = Path(old_entry['report'])
require(sha(p)==old_entry['report_sha256'], 'Old intermediate source changed')
r = read(p)
sa = read(OLD/'audit_snapshot_0058_0099.json')
require(r['sweep']==99 and sa['pass_replay'] and sha(p.parent/'pending_fields.npz')==old_entry['fields_sha256']==sa['fields_sha256'], 'Old intermediate replay/hash mismatch')
old_snapshot = record('L16 saved outer99',p,r,'fields_sha256','saved_intermediate')

current = newresult['current']
saved = [a for a in audit['rows'] if not a['final_load'] and a['index']==current['index'] and a['pass_replay'] and a['sweep']<=current['sweep']]
new_snapshot = None
if current['index'] not in completed and saved:
    a = max(saved,key=lambda a:a['sweep'])
    p = NEW/a['folder']/'report.json'
    r = read(p)
    require(abs(r['reaction']-a['reaction'])<1e-7 and sha(p.parent/'fields.npz')==r['fields_sha256'], 'L1 intermediate changed')
    new_snapshot = record('L1 in progress',p,r,'fields_sha256','in_progress')
    new_snapshot['sweep'] = r['sweep']

OUT.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
colors = {'FEM':'#465164','Direct':'#2864bd','L16 LBFGS':'#d1862c','L1 Adam':'#c43440'}
labels = {'FEM':'FEM: local mechanics + local damage','Direct':'Reference: local mechanics + nonlocal damage (direct)', 'L16 LBFGS':'Previous PINN: sigmoid / L16 / full-domain L-BFGS', 'L1 Adam':'Current PINN: sigmoid / L1 / random Adam'}
fig, axes = plt.subplots(1,2,figsize=(14,6.2))
for ax in axes:
    for label, rows in data.items():
        x = [r['load'] for r in rows]; y = [r['reaction'] for r in rows]
        kw = {'color':colors[label],'lw':2.,'label':labels[label]}
        if label in ['L1 Adam','L16 LBFGS']:
            kw.update(marker='o' if label=='L1 Adam' else 's',ms=4.5,ls='-' if label=='L1 Adam' else '--')
        ax.plot(x,y,**kw)
    ax.plot([old_rows[-1]['load'],old_snapshot['load']],[old_rows[-1]['reaction'],old_snapshot['reaction']],color=colors['L16 LBFGS'],lw=1.5,ls=':')
    ax.scatter([old_snapshot['load']],[old_snapshot['reaction']],s=58,marker='s',facecolors='white',edgecolors=colors['L16 LBFGS'],linewidths=1.8,zorder=8)
    if new_snapshot:
        ax.scatter([new_snapshot['load']],[new_snapshot['reaction']],s=75,marker='o',facecolors='white',edgecolors=colors['L1 Adam'],linewidths=2,zorder=9)
    ax.set(xlabel='Prescribed displacement U',ylabel='Reaction R',ylim=(-12,650))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.grid(alpha=.2)
axes[0].set(xlim=(0,.00603),title='Full loading path')
axes[1].set(xlim=(.0050,.00603),title='Crack-growth and unloading region')
axes[1].set_xticks([.0050,.0052,.0054,.0056,.0058,.0060])
axes[1].annotate('Direct drop\n0.005250 → 0.005275',xy=(.005275,1.59386849),xytext=(.00503,225),fontsize=9,color=colors['Direct'],arrowprops={'arrowstyle':'->','color':colors['Direct']})
axes[1].annotate('Current L1\n0.0058: R = 0.889',xy=(.0058,data['L1 Adam'][-1]['reaction']),xytext=(.00572,105),fontsize=9,color=colors['L1 Adam'],arrowprops={'arrowstyle':'->','color':colors['L1 Adam']})
# Post-fracture inset: show the otherwise visually indistinguishable residual forces.
ins = axes[1].inset_axes([.66,.58,.31,.32])
for label in ['FEM','Direct','L1 Adam']:
    points = [r for r in data[label] if r['load']>=.0057-1e-14]
    ins.plot([r['load'] for r in points],[r['reaction'] for r in points],color=colors[label],marker='o',ms=3,lw=1.4)
ins.scatter([old_snapshot['load']],[old_snapshot['reaction']],s=35,marker='s',facecolors='white',edgecolors=colors['L16 LBFGS'],linewidths=1.5,zorder=8)
if new_snapshot:
    ins.scatter([new_snapshot['load']],[new_snapshot['reaction']],s=40,marker='o',facecolors='white',edgecolors=colors['L1 Adam'],linewidths=1.5,zorder=9)
ins.set(xlim=(.00576,.00603),ylim=(0,1.6),title='Residual force')
ins.set_xticks([.0058,.0060]);ins.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));ins.tick_params(labelsize=8);ins.title.set_fontsize(9);ins.grid(alpha=.2)
handles, legends = axes[0].get_legend_handles_labels()
fig.legend(handles,legends,ncol=2,loc='upper center',bbox_to_anchor=(.5,.98),frameon=False,fontsize=10)
latest = f"L1 hollow circle: U={new_snapshot['load']:.4f}, saved outer {new_snapshot['sweep']} (load still running)." if new_snapshot else 'L1: independently audited completed loads only.'
fig.text(.5,.055,latest+'  L16 hollow square: U=0.0058, saved outer99 (not a completed load).',ha='center',fontsize=9)
fig.text(.5,.025,'PINN load endpoints are provisional. Different starting histories, early load grids and parameter scales; this is a protocol comparison.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.86))
fig.savefig(OUT/'force_displacement.png',dpi=180)
fig.savefig(OUT/'force_displacement.pdf')
plt.close(fig)
with (OUT/'curve_data.csv').open('w',newline='') as f:
    w = csv.DictWriter(f,fieldnames=['series','load','reaction','status']);w.writeheader()
    for label, rows in data.items():
        for row in rows:w.writerow({'series':label,**row})
    w.writerow({'series':'L16 saved outer99',**old_snapshot})
    if new_snapshot:w.writerow({k:v for k,v in {'series':'L1 in progress',**new_snapshot}.items() if k!='sweep'})
shutil.copy2(__file__,OUT/'plot_script.py')
payload = {'created_utc':STAMP,'same_geometry_and_material':True,'curves':data,'old_intermediate':old_snapshot,'new_intermediate':new_snapshot,'live_status_at_capture':{k:current[k] for k in ['load','sweep','reaction']},'source_manifest':manifest,'source_identity_sha256':{p.name:sha(p/'identity.json') for p in [NEW,OLD,FEM,REF,PREFIX]},'audit_complete_at_capture':audit['complete'],'full_certified_path':audit['full_certified_path'],'notes':['Mechanics local P1 in all four curves; FEM damage local P1, other damage nonlocal AT2.','Direct reference uses refined loads, plotted at actual displacement.','Previous L16 curve starts at certified physical0.0052 prefix; later endpoints provisional.','Current L1 starts physical zero; all positive load endpoints provisional.','Snapshots hollow; no extrapolation, optimizer change, FEM launch or training interruption.'],'figure_sha256':sha(OUT/'force_displacement.png')}
(OUT/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'latest_completed':data['L1 Adam'][-1],'in_progress_snapshot':new_snapshot,'captured_live':payload['live_status_at_capture']},indent=2))

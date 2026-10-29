"""Post-training physical-volume error concentration; no training reweighting."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.problem import load_reference,relative
p=Path(__file__).with_name('chmixed-analysis.json');d=json.loads(p.read_text())
for name,r in d['runs'].items():
    c=ConfigDict(json.loads((Path(r['run'])/'config.json').read_text()));ref=load_reference(c);x=ref['points'];v=ref['volumes']
    distance=np.abs(.5*min(c.geometry.axes)*(1-np.sum(((x-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes))**2,axis=1)))
    for cx,cy,rad in c.geometry.holes:distance=np.minimum(distance,np.abs(np.linalg.norm(x-np.array([cx,cy]),axis=1)-rad))
    mask=distance<c.nl.delta
    with np.load(Path(r['run'])/'cpu_audit_fields.npz') as f:phi=f['phi']
    err=(phi[-1]-ref['phi_nonlocal'][-1])**2*v
    r['boundary_audit']=dict(distance_rule='Outer defining-function distance approximation / exact circular distance, band < delta; diagnostic only',area_fraction=float(v[mask].sum()/v.sum()),terminal_native_error_squared_fraction=float(err[mask].sum()/err.sum()),terminal_native_band_l2=relative(phi[-1,mask],ref['phi_nonlocal'][-1,mask],v[mask]),terminal_native_bulk_l2=relative(phi[-1,~mask],ref['phi_nonlocal'][-1,~mask],v[~mask]))
    print(name,r['boundary_audit'],flush=True)
p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

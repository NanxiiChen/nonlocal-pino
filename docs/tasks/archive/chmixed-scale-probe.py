"""Known-IC physical scales; future reference fields used only for validation."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
from examples.confined_ch2d.configs.train_coarsening_global_mixed import get_config
from examples.confined_ch2d.problem import load_reference,initial_field
from examples.confined_ch2d.native_reference import native_matrix
c=get_config();r=load_reference(c);v=r['volumes'];A=native_matrix(r['points'],v,c)
phi0=initial_field(r['points'],c,xp=np);f=c.physical.beta*(phi0**3-phi0);g=c.physical.epsilon**2*(A@phi0);mu=f+g
rms=lambda a:np.sqrt(np.sum(a*a*v,axis=-1)/v.sum())
known={name:dict(rms=float(rms(a)),minimum=float(np.min(a)),maximum=float(np.max(a))) for name,a in [('phi0',phi0),('reaction0',f),('gradient0',g),('mu0',mu),('phi_t0',-c.physical.mobility*(A@mu))]}
report=dict(known_IC=known,chemical_output_scale=c.arch.chemical_scale,chemical_row_scale=c.loss.chemical_scale,
    validation_only=dict(time=r['time'].tolist(),phi_rms=rms(r['phi_nonlocal']).tolist(),mu_rms=rms(r['mu_nonlocal']).tolist()),
    scope='Only known_IC guides declared scales. Future reference RMS is diagnostic, never a training input.')
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')

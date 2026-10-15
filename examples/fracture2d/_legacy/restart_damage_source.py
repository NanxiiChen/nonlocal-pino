"""Hash-pinned certified neural predecessor for a new loading experiment."""
import json
from pathlib import Path
import equinox as eqx
import numpy as np
from examples.fracture2d._legacy.local_damage_trial import sha


def restore_certified_source(c,problem,mechanics):
    a=c.core.restart
    pins={a.source_fields:a.source_fields_sha256,a.source_network:a.source_network_sha256,
        a.source_report:a.source_report_sha256,a.source_parameterization:a.source_parameterization_sha256}
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError(f'certified restart source checksum differs:{path}')
    source_root=Path(a.source_fields).parent.parent
    identity=json.loads((source_root/'identity.json').read_text())
    audit=json.loads(Path(a.source_audit).read_text())
    report=json.loads(Path(a.source_report).read_text())
    record=next(row for row in audit['records'] if row['index']==a.initial_index)
    if (identity['geometry']!=problem.geometry or not audit['pass_replay'] or not record['ready']
            or not identity['own_zero_initial_state'] or identity['direct_targets'] or identity['damage_correction']
            or report['status']!='ready' or not report['outer_convergence']['converged']
            or report['index']!=a.initial_index or report['load']!=a.initial_index*c.data.load_step
            or record['state_sha256']!=a.source_fields_sha256 or record['network_sha256']!=a.source_network_sha256
            or report['state_sha256']!=a.source_fields_sha256 or report['network_sha256']!=a.source_network_sha256):
        raise ValueError('restart requires independently audited certified own-neural predecessor')
    with np.load(a.source_parameterization) as f:scale,gain=f['column_scale'].copy(),float(f['trunk_gain'])
    with np.load(Path(c.loading.output_dir)/'parameterization.npz') as f:
        np.testing.assert_array_equal(scale,f['column_scale']);assert gain==float(f['trunk_gain'])==problem.gain
    with np.load(a.source_fields) as f:fields={key:f[key].copy() for key in f.files}
    q,d,H=fields['q'],fields['d'],fields['H']
    if (d.shape!=(mechanics.n_material,) or H.shape!=d.shape or q.shape!=mechanics.load_direction.shape
            or not np.isfinite(q).all() or not np.isfinite(d).all() or not np.isfinite(H).all()):
        raise ValueError('certified restart fields have invalid shapes or nonfinite data')
    mechanics.factors(d)
    np.testing.assert_array_equal(H,np.maximum(fields['H_parent'],mechanics.driver(q)))
    np.testing.assert_array_equal(q[mechanics.constrained],report['load']*mechanics.load_direction[mechanics.constrained])
    force=mechanics.matrix(d)@q;reaction=float(force@mechanics.load_direction)
    np.testing.assert_allclose(reaction,report['reaction'],rtol=1e-9,atol=1e-6)
    limit=report['outer_convergence']['checks']['mechanics']['limit']
    if np.linalg.norm(force[mechanics.free])>limit*(1+1e-3):raise ValueError('certified restart force equilibrium fails')
    net=eqx.tree_deserialise_leaves(a.source_network,problem.initial)
    return fields,net,report,pins

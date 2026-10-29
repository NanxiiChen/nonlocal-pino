"""Generate a NEW native reference; explicitly reuse the unchanged local target."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT))
from examples.twophase2d.configs.train_vortex_nonlocal_flux import get_config
from examples.twophase2d.configs.train_vortex_conserved import get_config as baseline_config
from examples.twophase2d.problem import identity, load_reference, validate_config
from examples.twophase2d.solver import solve_reference, relative


def main():
    c = get_config(); validate_config(c)
    base = baseline_config()
    local = load_reference(base)
    previous_checks = json.loads(Path(base.reference.path).with_suffix('.json').read_text())
    assert previous_checks['status'] == 'passed'
    for name in ('local_time', 'local_space'):
        assert max(previous_checks['checks'][name].values()) <= c.reference.error_tolerance
    output = Path(c.reference.path)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    fields = []
    for dt in (c.reference.dt, c.reference.refined_dt):
        print(f'New native nonlocal-flux reference: n={c.sample.Nx}, dt={dt}', flush=True)
        fields.append(solve_reference(c, c.sample.Nx, 'nonlocal', dt))
    coarse, fine = fields
    checks = {}
    for field in ('phi', 'q'):
        checks[field+'_time'] = dict(
            pooled=relative(coarse[field], fine[field]),
            terminal=relative(coarse[field][-1], fine[field][-1]),
            max_snapshot=max(relative(a, b) for a, b in zip(coarse[field], fine[field], strict=True)))
        assert max(checks[field+'_time'].values()) <= c.reference.error_tolerance, checks
    mass = np.mean(fine['phi'], axis=(1, 2))
    checks['mass_max_drift'] = float(np.max(np.abs(mass-mass[0])))
    assert checks['mass_max_drift'] < 1e-10
    checks['native_vs_common_local_phi'] = dict(
        pooled=relative(fine['phi'], local['phi_local']),
        terminal=relative(fine['phi'][-1], local['phi_local'][-1]))
    # The native operator includes the selected first derivative; the shared
    # local target is copied by explicit path after its identity/refinement checks.
    np.savez_compressed(output, identity=json.dumps(identity(c), sort_keys=True),
        points=local['points'], time=local['time'],
        phi_local=local['phi_local'], q_local=local['q_local'],
        phi_nonlocal=fine['phi'], q_nonlocal=fine['q'],
        energy_local=local['energy_local'], energy_nonlocal=fine['energy'])
    record = dict(status='passed', identity=identity(c), checks=checks,
        method='Independent NumPy/SciPy IMEX-BDF2 with nonlocal first-derivative symbols; no training operator import',
        common_local_reference=base.reference.path,
        common_local_checks=previous_checks,
        configuration=c.reference.to_dict())
    output.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    load_reference(c)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()

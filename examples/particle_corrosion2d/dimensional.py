"""Explicit SI inputs and checked derived groups; legacy pilots stay readable."""
import numpy as np
from ml_collections import ConfigDict


SI_FIELDS = ('D', 'A', 'L', 'alpha_phi', 'omega_phi')
GEOMETRY_FIELDS = ('length', 'height', 'pit_x', 'pit_radius', 'mouth_half_width')


def has_dimensional_inputs(c):
    present = ('nondim' in c, 'geometry_si' in c,
               *(name in c.physical for name in SI_FIELDS))
    if any(present) and not all(present):
        raise ValueError('Dimensional particle configuration requires all SI fields, nondim and geometry_si')
    return all(present)


def derived_parameters(c):
    if not has_dimensional_inputs(c):
        raise ValueError('Cannot derive SI coefficients from a legacy dimensionless pilot')
    p, s = c.physical, c.nondim
    values = [p[name] for name in SI_FIELDS]+[s.L_c, s.T_c]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('SI material parameters, characteristic length and time must be positive finite')
    if p.kappa != 1.0:
        raise ValueError('SI physical.L is the actual matrix mobility; kappa must be1, never multiply L twice')
    groups = dict(n_ch=p.D*s.T_c/s.L_c**2, n_ac1=2*p.A*p.L*s.T_c,
                  n_ac2=p.L*p.omega_phi*s.T_c, n_ac3=p.L*p.alpha_phi*s.T_c/s.L_c**2)
    geometry = {name:c.geometry_si[name]/s.L_c for name in GEOMETRY_FIELDS}
    if not np.isfinite(list(geometry.values())).all() or min(geometry.values()) <= 0:
        raise ValueError('SI domain and pit dimensions must be positive finite')
    geometry['interface_k'] = np.sqrt(p.omega_phi/(2*p.alpha_phi))*s.L_c
    return groups, geometry


def derive_config(c):
    """Populate the cache consumed by existing PDE/geometry code during setup."""
    groups, geometry = derived_parameters(c)
    for name, value in groups.items():
        c.physical[name] = float(value)
    for name, value in geometry.items():
        c.geometry[name] = float(value)
    return c


def validate_dimensional(c):
    if not has_dimensional_inputs(c):
        return  # Explicit historical path: no SI interpretation was assigned.
    if c.sample.t_max != 1.0:
        raise ValueError('SI case requires tstar in[0,1]; change nondim.T_c to select physical duration')
    groups, geometry = derived_parameters(c)
    for target, expected, label in [(c.physical, groups, 'PDE'), (c.geometry, geometry, 'geometry')]:
        for name, value in expected.items():
            if not np.isclose(target[name], value, rtol=1e-12, atol=0):
                raise ValueError(f'Stale dimensional {label} field {name}; regenerate with derive_config')


def require_matching_units(saved, c):
    previous = ConfigDict(saved)
    validate_dimensional(previous)
    validate_dimensional(c)
    if has_dimensional_inputs(previous) != has_dimensional_inputs(c):
        raise ValueError('Input/reference dimensional parameterization differs from training')
    if has_dimensional_inputs(c):
        for name in ('nondim', 'geometry_si'):
            if previous[name].to_dict() != c[name].to_dict():
                raise ValueError(f'Input/reference dimensional {name} differs from training')

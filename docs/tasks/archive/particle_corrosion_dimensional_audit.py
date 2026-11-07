"""CPU-only forward/inverse unit audit; no reference solve or training.

Run from the repository root with python -m docs.tasks.archive.
particle_corrosion_dimensional_audit --output <receipt.json>.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from examples.corrosion2d.configs.train_base import get_config as corrosion_config
from examples.corrosion2d.residual_experiment import nondim
from examples.electropolish2d.configs.train_operator import get_config as electro_config
from examples.particle_corrosion2d.configs.train_online import get_config as particle_config
from examples.particle_corrosion2d.configs.train_speed2 import get_config as speed2_config
from examples.particle_corrosion2d.losses import residuals
from examples.particle_corrosion2d.solver import make_system


def groups(physical, length, time):
    return np.array([physical['D']*time/length**2,
                     2*physical['A']*physical['L']*time,
                     physical['L']*physical['omega_phi']*time,
                     physical['L']*physical['alpha_phi']*time/length**2])


def effective(config):
    p = config.physical
    return np.array([p.n_ch, p.kappa*p.n_ac1, p.kappa*p.n_ac2, p.kappa*p.n_ac3])


def energy_diffusion_anchor(physical, target, kappa):
    # Hold A,omega,alpha,D; solve length,time and actual matrix mobility.
    length = np.sqrt(physical['alpha_phi']/physical['omega_phi']*target[2]/target[3])
    time = target[0]*length**2/physical['D']
    actual = dict(physical, L=target[2]/(physical['omega_phi']*time))
    np.testing.assert_allclose(groups(actual, length, time), target, rtol=2e-15, atol=0)
    return dict(length_m=float(length), time_s=float(time), actual_matrix_L=actual['L'],
                base_L_before_kappa=actual['L']/kappa, physical=actual)


def main(output):
    jax.config.update('jax_enable_x64', True)
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('Use JAX_PLATFORMS=cpu for this scalar/synthetic audit')
    old, fast = particle_config(), speed2_config()
    target, fast_target = effective(old), effective(fast)
    np.testing.assert_array_equal(fast_target, 2*target)
    assert fast.physical.kappa == old.physical.kappa == 2
    assert fast.sample.t_max == old.sample.t_max == 1
    legacy = {}
    for name, config in [('electropolish', electro_config()), ('corrosion', corrosion_config())]:
        physical = config.physical.to_dict()
        calculated = groups(physical, config.nondim.L_c, config.nondim.T_c)
        np.testing.assert_allclose(calculated, nondim(config)[:4], rtol=2e-15, atol=0)
        legacy[name] = dict(physical=physical, scales=config.nondim.to_dict(),
                            effective_groups=calculated.tolist(),
                            interface_k=float(np.sqrt(physical['omega_phi']/(2*physical['alpha_phi']))*config.nondim.L_c))
    ep = legacy['electropolish']['physical']
    cp = legacy['corrosion']['physical']
    anchors = {}
    for name, physical in [('electropolish_energy_and_D', ep), ('corrosion_energy_and_D', cp)]:
        base = energy_diffusion_anchor(physical, target, old.physical.kappa)
        doubled = energy_diffusion_anchor(physical, fast_target, fast.physical.kappa)
        np.testing.assert_allclose(doubled['time_s'], 2*base['time_s'], rtol=1e-15)
        np.testing.assert_allclose(doubled['actual_matrix_L'], base['actual_matrix_L'], rtol=1e-15)
        anchors[name] = dict(original=base, speed2=doubled)
    length = legacy['electropolish']['scales']['L_c']
    time = target[0]*length**2/ep['D']
    fixed_length_physical = dict(ep, L=target[2]/(ep['omega_phi']*time),
                                alpha_phi=ep['omega_phi']*length**2*target[3]/target[2])
    np.testing.assert_allclose(groups(fixed_length_physical, length, time), target, rtol=2e-15)
    anchors['electropolish_length_D_A_omega'] = dict(length_m=length, original_time_s=time,
        speed2_time_s=2*time, physical=fixed_length_physical)

    # Compare production residuals with an independently dimensional SI oracle.
    chosen = anchors['electropolish_energy_and_D']['original']
    physical, lc, tc = chosen['physical'], chosen['length_m'], chosen['time_s']
    edge = np.array([[0,1.3,.2,0],[1.3,0,.4,.5],[.2,.4,0,.7],[0,.5,.7,0]], dtype=float)
    matrix = np.diag(edge.sum(axis=1))-edge
    points = np.array([[.5,0],[.5,.12],[.4,.13],[.7,.4]])
    mouth = np.array([True,False,False,False])
    core = np.arange(1,4)
    payload = dict(core_indices=jnp.asarray(core), neighbors=jnp.asarray(np.tile(np.arange(4),(3,1))),
                   coefficients=jnp.asarray(edge[core]))
    rng = np.random.default_rng(203)
    raw = rng.normal(size=(3,4,2)); dt = rng.normal(size=(3,4,2))
    phi = .5*(1+np.tanh(raw[...,0])); hp = 6*phi*(1-phi)
    gp = 2*phi*(1-phi)*(1-2*phi); dc = ep['c_se']-ep['c_le']
    conc = ep['c_le']*raw[...,1]+dc*phi**2*(3-2*phi)
    q = conc-dc*phi**2*(3-2*phi)-ep['c_le']
    physical_matrix = matrix/lc**2
    ac_si = dt[...,0]/tc+physical['L']*(physical['alpha_phi']*(physical_matrix@phi.T).T
        +physical['omega_phi']*gp-2*physical['A']*dc*q*hp)
    ch_si = dt[...,1]/tc+physical['D']*(physical_matrix@q.T).T
    ac, ch = (np.asarray(x) for x in residuals(jnp.asarray(raw),jnp.asarray(dt),payload,old))
    ac2, ch2 = (np.asarray(x) for x in residuals(jnp.asarray(raw),jnp.asarray(2*dt),payload,fast))
    np.testing.assert_allclose(ac,tc*ac_si[:,core].T,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(ch,tc*ch_si[:,core].T,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(ac2,2*ac,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(ch2,2*ch,rtol=1e-14,atol=1e-14)
    rhs,jac,_,_ = make_system(old,sparse.csr_matrix(matrix),points,mouth)
    rhs2,jac2,_,_ = make_system(fast,sparse.csr_matrix(matrix),points,mouth)
    state = np.r_[phi[0,core],conc[0,core]]
    np.testing.assert_allclose(rhs2(.31,state),2*rhs(.62,state),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(jac2(.31,state).toarray(),2*jac(.62,state).toarray(),rtol=1e-14,atol=1e-14)

    run_root = Path('/root/autodl-tmp/tf-logs/particle_corrosion2d')
    saved = []
    for name,expected in [('20261106-online-geometry-spatial-w192-cold20000-02',old),
        ('20261106-speed2-online-w192-cold20000-01',fast),
        ('20261106-speed2-mouth-nodes-w192-cold20000-01',fast),
        ('20261106-speed2-time32-w192-cold20000-01',fast)]:
        config = json.loads((run_root/name/'config.json').read_text())
        assert config['physical'] == expected.physical.to_dict()
        assert config['sample']['t_max'] == 1
        saved.append(str(run_root/name))
    invariant = target[3]/target[0]
    result = dict(status='passed', group_order=['N_CH','effective_N_AC1','effective_N_AC2','effective_N_AC3'],
        legacy=legacy, particle_original=target.tolist(), particle_speed2=fast_target.tolist(),
        invariant_particle=float(invariant), invariant_electropolish=ep['L']*ep['alpha_phi']/ep['D'],
        invariant_ratio_electropolish_over_particle=ep['L']*ep['alpha_phi']/ep['D']/invariant,
        anchors=anchors, same_electropolish_L_time_from_AC_only=target[2]/(ep['L']*ep['omega_phi']),
        same_electropolish_material_at_5s=groups(ep,length,5).tolist(),
        same_electropolish_material_at_10s=groups(ep,length,10).tolist(),
        si_residual_max_abs_difference=dict(ac=float(np.max(np.abs(ac-tc*ac_si[:,core].T))),
                                            ch=float(np.max(np.abs(ch-tc*ch_si[:,core].T)))),
        scaling_checks=dict(residual_with_doubled_time_derivative=True,solver_rhs=True,solver_jacobian=True),
        saved_run_physics_checked=saved,
        interpretation='No exact length/time pair makes all current particle groups equal to the electropolish material tuple. Anchors explicitly permit different dimensional mobility, or mobility and alpha. No physical coefficients or training defaults changed.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','particle_original','particle_speed2',
        'invariant_ratio_electropolish_over_particle','si_residual_max_abs_difference','scaling_checks']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)

"""Accepted own-zero PINN defaults; all comparisons are explicit overrides."""
from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.runtime = ConfigDict(dict(platform='gpu'))
    c.physical = ConfigDict(dict(E=210000., nu=.3, Gc=2.7, l0=.024, k_res=1e-8))
    c.geometry = ConfigDict(dict(grip_extent=.048, h_grip=.004))
    c.nl = ConfigDict(dict(delta_d=.008, visibility_tolerance=1e-12))
    c.data = ConfigDict(dict(
        cloud='/root/autodl-tmp/data/fracture2d/scale_separation_20261001/cloud_h004_api.npz',
        basis_path='/root/autodl-tmp/tf-logs/fracture2d/diagnostics/compact_damage_polish_fixedH_20261011/basis.npz'))
    c.mesh = ConfigDict(dict(
        prepared_dir='/root/autodl-tmp/data/fracture2d/geometry_assets',
        prepare=False))
    c.arch = ConfigDict(dict(type='mlp', hidden_dim=96, num_layers=3,
        activation='swish', use_spatial_emb=False, spatial_emb_dim=64,
        spatial_scale=16., tensor_split=False, multi_head=False,
        film_hidden_dim=16, film_num_layers=1, output_head='sigmoid',
        initial_damage=1e-3, trunk_gain=1.))
    c.features = ConfigDict(dict(tip=[.5, .5], powers=[0, 1, 2, 4, 8, 16]))
    c.loss = ConfigDict(dict(norm='l1', floor=1e-15, weight_mode='uniform',
        weight_boost=9., weight_power=2., weight_epsilon=1e-3, weight_cap=50.))
    c.sample = ConfigDict(dict(mode='full', core_count=2048,
        pool_size=2048, resample_every=1, seed=1729,
        start=[.5, .5], end=[1., .5], width=.024, boost=9., power=2.,
        tip_width=.024, tip_boost=0., tip_power=2.))
    c.train = ConfigDict(dict(seed=0, inner_steps=128, optimizer='adam',
        learning_rate=1e-5, decay_steps=512, decay_rate=1., minimum_rate=1e-5,
        grad_clip=1000., b1=.9, b2=.999, eps=1e-8, eps_root=0.,
        shampoo_beta=.95, correct_bias=True, precondition_frequency=5,
        max_precond_dim=128, precondition_1d=False, weight_decay=0.,
        lbfgs=ConfigDict(dict(maxiter=128, maxfun=2048, memory=20,
            max_linesearch=30, gtol=1e-12, ftol=1e-15))))
    c.loading = ConfigDict(dict(du=1e-4, u_final=.0060, critical_load=.0053,
        prefix_outer_steps=128, outer_steps=1024, omega=1., field_stop=True,
        damage_tolerance=1e-5, displacement_tolerance=1e-4,
        minimum_sweeps=3, consecutive_passes=3, advance_on_budget=True))
    c.validation = ConfigDict(dict(cg_rtol=1e-9, cg_atol=1e-13,
        cg_maxiter=10000, roundoff_factor=1024., range_atol=1e-3))
    c.output = ConfigDict(dict(
        root='/root/autodl-tmp/tf-logs/fracture2d/runs/pinn', resume=False, event_dir='',
        reference_root='/root/autodl-tmp/data/fracture2d/nonlocal_reference',
        tensorboard=True, figure_every=1, figure_dpi=110, scalar_every=8,
        checkpoint_sweeps=[1, 8, 32, 64, 128, 256, 512, 1024],
        cpu_threads=4))
    return c

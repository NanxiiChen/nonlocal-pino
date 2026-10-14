"""Morphology-first L16/full/L-BFGS at fixed .0053 with common .0052 parent."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.damage_norm_trial import get_config as parent_config


def get_config():
    c=parent_config()
    c.output_dir='/root/autodl-tmp/tf-logs/fracture2d/norm_optimizer_coupled_L16_20261014'
    c.cases=[dict(name='pinn_full',solver='pinn',omega=1.,max_sweeps=644)]
    c.checkpoints=[1,8,32,64,128,192,256,384,512,644]
    c.coupled_training=ConfigDict(dict(norm='l16',optimizer='lbfgs',power=16,floor=1e-15,
        inner_updates=128,sampling='all_nodes',stop_early=False,
        selection='user_crack_morphology_priority',network_initialization='original_preceding_load_parent',
        lbfgs=dict(maxiter=128,maxfun=8192,memory=20,max_linesearch=50,gtol=1e-12,ftol=0.),
        screen_root='/root/autodl-tmp/tf-logs/fracture2d/norm_optimizer_damage_20261014',
        screen_pins=ConfigDict({
            'identity.json':'848aa0448564ed4a54acbe486bff3a7be2b43f2c1a2c4c1acdb684ca3f6e8993',
            'result.json':'16ceaca0e364f3302331509a9bc6249f382ea314de8f7266f4dd5cfe76e2a2cc',
            'selection.json':'dae9b93f3a3d659e72ad054160493a3cffed5739062d7c1bd62cedd3619d739a',
            'independent_review/audit.json':'7e5a1d9246e4fead5030906efc47796da84105a6dfb29bea09b709518e9cd302'
        },allow_dotted_keys=True)),allow_dotted_keys=True)
    return c

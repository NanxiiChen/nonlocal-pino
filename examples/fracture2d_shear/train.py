"""History-conditioned local FEM-energy shear PINN; ml_collections CLI."""
from pathlib import Path
from examples.fracture2d.loading import run


def run_training(config):
    if 'boundary' not in config or config.boundary.mode != 'horizontal_clamped_edges':
        raise ValueError('shear example requires horizontal_clamped_edges')
    return run(config, damage_solver='pinn')


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'train_base.py'),
                                   'Shear PINN configuration', lock_config=True)
    def execute(_):
        run_training(flags.FLAGS.config)
    app.run(execute)


if __name__ == '__main__':
    main()

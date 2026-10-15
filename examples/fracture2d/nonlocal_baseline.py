"""Local P1 direct mechanics + nonlocal AT2 direct damage, matched PINN scheme."""
from pathlib import Path


def run_baseline(config):
    from examples.fracture2d.loading import run
    return run(config, damage_solver='direct')


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config',
        str(Path(__file__).with_name('configs')/'nonlocal_base.py'),
        'Fracture2D direct configuration', lock_config=True)
    def _main(_):
        run_baseline(flags.FLAGS.config)
    app.run(_main)


if __name__ == '__main__':
    main()

"""Matched direct local P1 mechanics and AT2 damage for the L-panel."""
from pathlib import Path
from examples.fracture2d.loading import run


def run_reference(config):
    if 'boundary' not in config or config.boundary.mode != 'l_shaped_panel':
        raise ValueError('L-panel reference requires l_shaped_panel boundary data')
    return run(config, damage_solver='direct')


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'solver_base.py'),
                                   'L-panel FEM reference configuration', lock_config=True)
    def execute(_):
        run_reference(flags.FLAGS.config)
    app.run(execute)


if __name__ == '__main__':
    main()

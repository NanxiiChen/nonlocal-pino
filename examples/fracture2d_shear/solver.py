"""Matched local P1 FEM shear reference; direct displacement and damage."""
from pathlib import Path
from examples.fracture2d.loading import run


def run_reference(config):
    if 'boundary' not in config or config.boundary.mode != 'horizontal_clamped_edges':
        raise ValueError('shear reference requires horizontal_clamped_edges')
    return run(config, damage_solver='direct')


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'solver_base.py'),
                                   'Shear FEM reference configuration', lock_config=True)
    def execute(_):
        run_reference(flags.FLAGS.config)
    app.run(execute)


if __name__ == '__main__':
    main()

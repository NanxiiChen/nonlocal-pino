"""Independent UFL L-panel validation using the common material-edge checks."""
from pathlib import Path
from examples.fracture2d_shear.validate_fem import validate


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'train_base.py'),
                                   'Independent L-panel FEM validation', lock_config=True)
    def execute(_):
        c = flags.FLAGS.config
        if c.boundary.mode != 'l_shaped_panel':
            raise ValueError('L-panel validation requires l_shaped_panel')
        validate(c)
    app.run(execute)


if __name__ == '__main__':
    main()

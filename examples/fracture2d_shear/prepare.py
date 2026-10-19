"""Prepare material-only shear mesh/BC data without a nonlocal operator."""
from pathlib import Path
from examples.fracture2d.mesh import prepare


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'train_base.py'),
                                   'Shear geometry configuration', lock_config=True)
    def execute(_):
        c = flags.FLAGS.config
        if 'boundary' not in c or c.boundary.mode != 'horizontal_clamped_edges':
            raise ValueError('shear preparation requires horizontal_clamped_edges')
        Path(c.mesh.prepared_dir).parent.mkdir(parents=True, exist_ok=True)
        prepare(c)
    app.run(execute)


if __name__ == '__main__':
    main()

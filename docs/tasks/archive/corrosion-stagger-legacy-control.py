"""Replay archived joint step with current evaluation, preserving both sources."""
import ast
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from examples.corrosion2d import residual_experiment as runner

ORIGINAL = Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-residual-phase-control-seed42')
NAME = '20261025-stagger-diagnostic-legacy-joint-phase-seed42'

def main():
    source = (ORIGINAL/'source/examples/corrosion2d/residual_experiment.py').read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_step')
    code = ast.get_source_segment(source, node)
    namespace = {'eqx': eqx, 'jax': jax, 'jnp': jnp}
    exec(code, namespace)
    def make_step(loss_vector, optimizer, config):
        if config.loss.stagger_steps != 0:
            raise ValueError('Legacy diagnostic only supports joint updates')
        legacy = namespace['make_step'](loss_vector, optimizer, config)
        def step(net, state, weights, times, x, sub_data, mask, update_weights, active):
            return legacy(net, state, weights, times, x, sub_data, mask, update_weights)
        return step
    runner.make_step = make_step
    config = runner.get_config('corrosion2d', 'phase', 42)
    config.log.run_name = NAME
    runner.train(config)
    run = Path(config.log.log_dir)/NAME
    (run/'legacy_make_step.py').write_text(code+'\n')
    (run/'diagnostic_provenance.json').write_text(json.dumps({
        'archived_runner': str(ORIGINAL/'source/examples/corrosion2d/residual_experiment.py'),
        'override': 'make_step extracted by AST; wrapper discards all-true active; current runner/evaluation otherwise',
        'controller': str(Path(__file__).resolve()),
    }, indent=2)+'\n')

if __name__ == '__main__':
    main()

"""Record numerical execution context without changing it or exposing unrelated env."""
import os
import platform
import shlex

import equinox
import jax
import jaxlib
import numpy
import optax


EXECUTION_ENVIRONMENT = (
    'XLA_FLAGS', 'JAX_ENABLE_COMPILATION_CACHE', 'JAX_COMPILATION_CACHE_DIR',
    'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
    'JAX_DEFAULT_MATMUL_PRECISION', 'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG',
)


def configure_execution(c):
    """Apply only explicitly selected, qualified compilation controls before JAX init."""
    if 'autotune_level' not in c.train:
        return
    if isinstance(c.train.autotune_level,bool) or c.train.autotune_level!=0:
        raise ValueError('The fixed-adjoint execution control requires autotune_level=0')
    flags=os.environ['XLA_FLAGS'] if 'XLA_FLAGS' in os.environ else ''
    matches=[v for v in shlex.split(flags) if v.startswith('--xla_gpu_autotune_level=')]
    if any(v!='--xla_gpu_autotune_level=0' for v in matches):
        raise ValueError('Inherited XLA autotuning conflicts with train.autotune_level=0')
    if not matches:
        os.environ['XLA_FLAGS']=(flags+' --xla_gpu_autotune_level=0').strip()


def execution_provenance():
    """Call after the explicit configuration/backend setup, before optimization."""
    return dict(
        versions=dict(python=platform.python_version(), jax=jax.__version__,
                      jaxlib=jaxlib.__version__, equinox=equinox.__version__,
                      optax=optax.__version__, numpy=numpy.__version__),
        environment={name: os.environ[name] if name in os.environ else None
                     for name in EXECUTION_ENVIRONMENT},
        effective_jax=dict(enable_x64=jax.config.jax_enable_x64,
                           matmul_precision=jax.config.jax_default_matmul_precision,
                           compilation_cache=jax.config.jax_enable_compilation_cache,
                           compilation_cache_dir=jax.config.jax_compilation_cache_dir),
        devices=[dict(id=device.id, platform=device.platform, kind=device.device_kind)
                 for device in jax.devices()],
    )

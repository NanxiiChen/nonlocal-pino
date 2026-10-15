"""Cache original support rows once; bulk-remap genuinely fresh core batches."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.core_halo_damage import pack_rows


@dataclass(frozen=True)
class SamplingRows:
    columns: np.ndarray
    values: np.ndarray


def make_sampling_rows(operator, *, row_width):
    """Validate/canonicalize the full operator once, before any batch/JIT loop."""
    D = csr_matrix(operator, copy=True)
    if D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError('sampling support requires a nonempty square operator')
    columns, values = pack_rows(D, width=row_width)
    columns.flags.writeable = False
    values.flags.writeable = False
    return SamplingRows(columns, values)


def sample_core_halo(rows, core, *, node_capacity):
    """Gather cached rows and remap the exact halo without per-core Python loops.

    Repeated cores retain their original order/multiplicity. Zero-coefficient
    padding never expands the actual halo; insufficient capacity fails loudly.
    """
    ids = np.asarray(core)
    if (not isinstance(rows, SamplingRows) or ids.ndim != 1 or ids.size == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or ids.min() < 0 or ids.max() >= rows.columns.shape[0]
            or type(node_capacity) is not int or node_capacity < 1):
        raise ValueError('cached support needs valid core IDs and explicit positive halo capacity')
    neighbors, values = rows.columns[ids], rows.values[ids]
    active = values != 0
    halo, inverse = np.unique(np.concatenate((ids, neighbors[active])), return_inverse=True)
    if halo.size > node_capacity:
        raise ValueError('cached core+halo capacity would truncate original support')
    global_ids = np.zeros(node_capacity, np.int32)
    global_ids[:halo.size] = halo
    columns = np.zeros(neighbors.shape, np.int32)
    columns[active] = inverse[ids.size:]
    return dict(global_ids=global_ids, node_mask=np.arange(node_capacity) < halo.size,
                core_local=inverse[:ids.size].astype(np.int32), columns=columns, values=values)

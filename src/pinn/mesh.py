"""Gmsh .msh file reader for importing point clouds into PINN training.

Supports Gmsh ASCII format 4.1 and format 2.
Only the $Nodes section is parsed; element data is ignored.
"""

import jax.numpy as jnp


def read_gmsh_nodes(path, *, scale=None):
    """Parse node coordinates from a Gmsh .msh file (ASCII format 4.x or 2.x).

    Args:
        path: path to the .msh file.
        scale: optional scalar or (d,) array to multiply coordinates by.

    Returns:
        (N, dim) JAX array of node coordinates.
    """

    with open(path) as f:
        lines = f.readlines()

    version = _detect_version(lines)
    if version is None:
        raise ValueError(f"Not a valid Gmsh ASCII file: {path}")

    if version >= 4:
        nodes = _parse_nodes_v4(lines)
    else:
        nodes = _parse_nodes_v2(lines)

    nodes = jnp.array(nodes)
    if scale is not None:
        nodes = nodes * jnp.array(scale)
    return nodes


def _detect_version(lines):
    for i, line in enumerate(lines):
        if line.strip() == "$MeshFormat":
            parts = lines[i + 1].strip().split()
            return float(parts[0])
    return None


def _parse_nodes_v2(lines):
    """Parse $Nodes section in Gmsh v2 format."""
    import numpy as np
    i = _find_section(lines, "$Nodes")
    if i is None:
        raise ValueError("No $Nodes section found")
    n_nodes = int(lines[i + 1].strip())
    nodes = []
    for j in range(i + 2, i + 2 + n_nodes):
        parts = lines[j].strip().split()
        if parts:  # skip empty lines
            nodes.append([float(x) for x in parts[1:]])
    return np.array(nodes)


def _parse_nodes_v4(lines):
    """Parse $Nodes section in Gmsh v4.1 format.

    Each entity block is::

        entityDim entityTag parametric numNodesInBlock
        nodeTag₁           ┐
        ...                ├─ numNodesInBlock tag lines
        nodeTagₙ           ┘
        x₁ y₁ z₁           ┐
        ...                ├─ numNodesInBlock coordinate lines
        xₙ yₙ zₙ           ┘
    """
    import numpy as np
    i = _find_section(lines, "$Nodes")
    if i is None:
        raise ValueError("No $Nodes section found")

    header = lines[i + 1].strip().split()
    n_entity_blocks = int(header[0])
    n_nodes = int(header[1])

    tag_to_coords = {}
    pos = i + 2

    for _ in range(n_entity_blocks):
        block = lines[pos].strip().split()
        pos += 1
        parametric = int(block[2])
        n_in_block = int(block[3])

        if parametric != 0:
            raise ValueError(f"Parametric nodes not supported (line {pos})")

        # Read node tags
        tags = []
        for j in range(n_in_block):
            tags.append(int(lines[pos + j].strip().split()[0]))
        pos += n_in_block

        # Read coordinates
        for j in range(n_in_block):
            parts = lines[pos + j].strip().split()
            tag_to_coords[tags[j]] = [float(x) for x in parts]
        pos += n_in_block

    nodes = np.array([tag_to_coords[t] for t in sorted(tag_to_coords)])
    assert len(nodes) == n_nodes, f"Expected {n_nodes} nodes, got {len(nodes)}"
    return nodes


def _find_section(lines, name):
    for i, line in enumerate(lines):
        if line.strip() == name:
            return i
    return None

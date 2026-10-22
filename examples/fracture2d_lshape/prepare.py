"""Gmsh material-only L-panel with an exactly resolved finite loading patch."""
from pathlib import Path
import numpy as np
from examples.fracture2d.mesh import l_panel_boundary, prepare


def make_cloud(c):
    import gmsh
    if c.boundary.mode != 'l_shaped_panel' or not np.isfinite(c.geometry.mesh_size) or c.geometry.mesh_size <= 0:
        raise ValueError('explicit L-panel mode and positive mesh size required')
    path = Path(c.data.cloud)
    if path.exists():
        raise ValueError(f'refusing to overwrite L-panel cloud: {path}')
    b = c.boundary
    if (not np.isfinite([c.geometry.width, c.geometry.height, b.corner_x,
                        b.corner_y, b.load_start, b.load_end]).all()
            or not 0 < b.corner_x < c.geometry.width
            or not 0 < b.corner_y < c.geometry.height
            or not b.corner_x < b.load_start < b.load_end <= c.geometry.width):
        raise ValueError('invalid explicit L-panel dimensions/load patch')
    vertices = [[0., 0.], [b.corner_x, 0.], [b.corner_x, b.corner_y],
                [b.load_start, b.corner_y], [b.load_end, b.corner_y]]
    if b.load_end < c.geometry.width:
        vertices.append([c.geometry.width, b.corner_y])
    vertices.extend([[c.geometry.width, c.geometry.height], [0., c.geometry.height]])
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.NumThreads', c.output.cpu_threads)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.option.setNumber('Mesh.MeshSizeMin', c.geometry.mesh_size)
        gmsh.option.setNumber('Mesh.MeshSizeMax', c.geometry.mesh_size)
        gmsh.model.add('l_panel')
        points = [gmsh.model.geo.addPoint(x, y, 0., c.geometry.mesh_size) for x, y in vertices]
        lines = [gmsh.model.geo.addLine(p, q) for p, q in zip(points, points[1:]+points[:1], strict=True)]
        loop = gmsh.model.geo.addCurveLoop(lines)
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        sorted_tags = tags[order]
        coordinates = xyz.reshape(-1, 3)[order, :2].astype(np.float64)
        _, connectivity = gmsh.model.mesh.getElementsByType(2)
        indices = np.searchsorted(sorted_tags, connectivity)
        if np.any(indices >= len(sorted_tags)) or not np.array_equal(sorted_tags[indices], connectivity):
            raise ValueError('Gmsh triangle node tags are missing')
        triangles = indices.reshape(-1, 3).astype(np.int64)
    finally:
        gmsh.finalize()
    vertices = coordinates[triangles]
    edge_a, edge_b = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    areas = .5*abs(edge_a[:, 0]*edge_b[:, 1]-edge_a[:, 1]*edge_b[:, 0])
    volumes = np.bincount(triangles.ravel(), weights=np.repeat(areas/3, 3),
                          minlength=len(coordinates)).astype(np.float64)
    tol = 1e-12*max(1., c.geometry.width, c.geometry.height)
    bottom = abs(coordinates[:, 1]) <= tol
    loaded = ((abs(coordinates[:, 1]-b.corner_y) <= tol)
              & (coordinates[:, 0] >= b.load_start-tol) & (coordinates[:, 0] <= b.load_end+tol))
    cloud = dict(coordinates=coordinates, triangles=triangles, volumes=volumes,
                 bottom=bottom, top=loaded, anchor_particle=np.int64(np.flatnonzero(bottom)[0]))
    l_panel_boundary(cloud, c)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cloud)
    return cloud


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'train_base.py'),
                                   'L-panel mesh configuration', lock_config=True)
    def execute(_):
        c = flags.FLAGS.config
        if Path(c.mesh.prepared_dir).exists():
            raise ValueError('refusing to overwrite a prepared L-panel geometry')
        make_cloud(c)
        Path(c.mesh.prepared_dir).parent.mkdir(parents=True, exist_ok=True)
        prepare(c)
    app.run(execute)


if __name__ == '__main__':
    main()

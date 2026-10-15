"""Standalone Gmsh worker; run with the installed Gmsh Python interpreter.

Size overrides are applied after opening the existing benchmark geometry,
because its DefineNumber assignments overwrite parser constants from the CLI.
This worker imports no FEM or JAX solver.
"""

import argparse
from pathlib import Path

import gmsh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--h-far", type=float, required=True)
    parser.add_argument("--h-near", type=float, required=True)
    args = parser.parse_args()
    if not args.geo.is_file() or not args.mesh.parent.is_dir() or args.mesh.exists():
        raise ValueError("Gmsh worker input/output ownership check failed")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.open(str(args.geo))
        if set(gmsh.model.getEntities(0)) != {(0, index) for index in range(1, 11)}:
            raise ValueError("Gmsh worker requires the declared ten-point rounded-notch benchmark")
        gmsh.model.mesh.setSize([(0, index) for index in range(1, 5)], args.h_far)
        gmsh.model.mesh.setSize([(0, index) for index in range(5, 11)], args.h_near)
        gmsh.option.setNumber("Mesh.MeshSizeMin", args.h_near)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.h_far)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(args.mesh))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()

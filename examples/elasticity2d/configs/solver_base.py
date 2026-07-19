"""FEM solver config for 2D elastic plate with hole."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Geometry (non-dimensional — steel plate 2m×1.5m, L_c=2m)
    c.geometry = config_dict.ConfigDict()
    c.geometry.Lx = 1.0        # Lx* = 2m / 2m
    c.geometry.Ly = 0.75       # Ly* = 1.5m / 2m
    c.geometry.a = 0.15        # a* = 0.3m / 2m

    # Material (non-dimensional, ν=0.3)
    c.physical = config_dict.ConfigDict()
    c.physical.E = 1.0         # E* = 1
    c.physical.nu = 0.3
    c.physical.lam = 0.5769    # λ* = ν/((1+ν)(1-2ν))
    c.physical.mu = 0.3846     # μ* = 1/(2(1+ν))

    # Loading (2mm displacement, u_c=2mm → u_load*=1.0)
    c.load = config_dict.ConfigDict()
    c.load.u_load = 1.0        # u_load* = 2mm / 2mm

    # Mesh
    c.mesh = config_dict.ConfigDict()
    c.mesh.geo_file = "examples/elasticity2d/mesh.geo"
    c.mesh.msh_file = "examples/elasticity2d/mesh.msh"

    # Output
    c.output = config_dict.ConfigDict()
    c.output.dir = "/root/autodl-tmp/data/elasticity2d"
    c.output.filename = "results.npz"

    return c

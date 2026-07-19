"""FEM solver config for 2D elastic plate with hole."""

from ml_collections import config_dict


def get_config():
    c = config_dict.ConfigDict()

    # Geometry
    c.geometry = config_dict.ConfigDict()
    c.geometry.Lx = 1.5        # plate length in x
    c.geometry.Ly = 1.0        # plate length in y
    c.geometry.a = 0.2         # hole radius (centred at Lx/2, Ly/2)

    # Material (non-dimensional: E*=1, nu=0.3)
    c.physical = config_dict.ConfigDict()
    c.physical.E = 1.0         # Young's modulus
    c.physical.nu = 0.3        # Poisson's ratio (plane strain)
    c.physical.lam = 0.5769    # λ = Eν/((1+ν)(1-2ν)) for ν=0.3
    c.physical.mu = 0.3846     # μ = E/(2(1+ν)) for ν=0.3

    # Loading (displacement-controlled)
    c.load = config_dict.ConfigDict()
    c.load.u_load = 0.1        # prescribed u_x on right edge

    # Mesh
    c.mesh = config_dict.ConfigDict()
    c.mesh.geo_file = "examples/elasticity2d/mesh.geo"
    c.mesh.msh_file = "examples/elasticity2d/mesh.msh"

    # Output
    c.output = config_dict.ConfigDict()
    c.output.dir = "/root/autodl-tmp/data/elasticity2d"
    c.output.filename = "results.npz"

    return c

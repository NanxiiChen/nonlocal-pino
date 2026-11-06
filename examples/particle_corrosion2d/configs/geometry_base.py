"""Dimensionless, smooth irregular inclusions in the fixed pitting domain."""
from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.geometry = ConfigDict(dict(length=1.0, height=0.6, pit_x=0.5,
        pit_radius=0.12, mouth_half_width=0.055, interface_k=30.0))
    c.particles = ConfigDict(dict(count=8, placement='near_first',
        equivalent_radius_range=(0.045, 0.075), aspect_range=(1.0, 1.45),
        center_x_range=(0.12, 0.88), center_y_range=(0.09, 0.48),
        modes=(3, 4, 5), coefficient_std=(0.07, 0.035, 0.018),
        coefficient_max=(0.14, 0.07, 0.036),
        wall_clearance=0.025, particle_clearance=0.035, pit_clearance=0.03,
        minimum_curvature_radius=0.015,
        # Known-geometry prior: ensure some particles are within reach of the pit.
        near_pit_distance=0.12, minimum_near_pit_count=4,
        max_particle_attempts=1000, max_geometry_attempts=100))
    c.validation = ConfigDict(dict(boundary_points=512, refined_points=1024,
        area_points=4096, area_relative_tolerance=1e-10,
        gap_refinement_tolerance=0.0015, curvature_relative_tolerance=0.005,
        membership_tolerance=1e-12))
    return c

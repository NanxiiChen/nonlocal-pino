"""Geometry helpers for the rounded-slot fracture2d benchmark."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoundedSlot:
    """Finite-width slot opening at the left boundary and capped on the right."""

    tip_x: float
    center_y: float
    width: float
    radius: float

    @property
    def center_x(self):
        return self.tip_x - self.radius

    @property
    def tip(self):
        return np.asarray([self.tip_x, self.center_y], dtype=float)

    @property
    def center(self):
        return np.asarray([self.center_x, self.center_y], dtype=float)


def make_rounded_slot(tip_x, center_y, width, radius):
    """Construct and validate the rounded-slot geometry parameters."""
    values = {
        "tip_x": tip_x,
        "center_y": center_y,
        "width": width,
        "radius": radius,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"Rounded-slot parameters must be finite, got {values}")
    if tip_x <= 0.0 or center_y <= 0.0 or width <= 0.0 or radius <= 0.0:
        raise ValueError(f"Rounded-slot parameters must be positive in-domain, got {values}")
    if not np.isclose(width, 2.0 * radius, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "The selected semicircular tip requires width == 2 * radius, "
            f"got width={width} and radius={radius}"
        )
    if tip_x <= radius:
        raise ValueError("The rounded tip must leave a positive slot length")
    return RoundedSlot(
        tip_x=float(tip_x),
        center_y=float(center_y),
        width=float(width),
        radius=float(radius),
    )


def points_inside_slot(points, slot, include_boundary=False, tolerance=1.0e-12):
    """Return a mask for points in the rectangular slot plus right semicircle."""
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 2:
        raise ValueError(f"points must have trailing shape (2,), got {points.shape}")

    x = points[..., 0]
    y = points[..., 1]
    dx = x - slot.center_x
    dy = y - slot.center_y
    if include_boundary:
        rectangle = (x <= slot.center_x + tolerance) & (np.abs(dy) <= slot.radius + tolerance)
        cap = (
            (x > slot.center_x - tolerance)
            & (x <= slot.tip_x + tolerance)
            & (dx * dx + dy * dy <= slot.radius**2 + tolerance)
        )
    else:
        rectangle = (x <= slot.center_x + tolerance) & (np.abs(dy) < slot.radius - tolerance)
        cap = (
            (x > slot.center_x + tolerance)
            & (x < slot.tip_x - tolerance)
            & (dx * dx + dy * dy < slot.radius**2 - tolerance)
        )
    return rectangle | cap

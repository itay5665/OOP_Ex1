"""Geometric checks for triangles and trapezoids.

``EPS`` is the length tolerance from the assignment Q&A. ``ANGLE_TOL`` is the
absolute sine used for collinearity and coplanarity. Both are 0.001.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projection.plane import as_point

EPS = 0.001
ANGLE_TOL = 0.001


@dataclass(frozen=True)
class ValidationResult:
    """``ok`` is true when every rule passed. ``reason`` is empty in that case."""

    ok: bool
    reason: str = ""

    @staticmethod
    def success() -> ValidationResult:
        return ValidationResult(True, "")

    @staticmethod
    def failure(reason: str) -> ValidationResult:
        return ValidationResult(False, reason)


def _distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def validate_triangle(a, b, c) -> ValidationResult:
    """Check the Q&A triangle rules plus degeneracy and collinearity."""
    a = as_point(a)
    b = as_point(b)
    c = as_point(c)
    dab = _distance(a, b)
    dac = _distance(a, c)
    dbc = _distance(b, c)
    if dab < EPS or dac < EPS:
        short = dab if dab < dac else dac
        return ValidationResult.failure(
            f"degenerate triangle: leg length {short:.4f} < EPS"
        )
    diff = abs(dab - dac)
    if diff >= EPS:
        return ValidationResult.failure(
            f"invalid plane — |ab|-|ac| = {diff:.4f} >= EPS"
        )
    if dbc < EPS:
        return ValidationResult.failure(
            f"degenerate triangle: |bc| = {dbc:.4f} < EPS"
        )
    cross = np.cross(b - a, c - a)
    sine = float(np.linalg.norm(cross) / (dab * dac))
    if sine < ANGLE_TOL:
        return ValidationResult.failure(
            f"degenerate triangle: points are collinear (|sin| = {sine:.4f})"
        )
    return ValidationResult.success()


def validate_trapezoid(a1, a2, b, c) -> ValidationResult:
    """Check equal legs, base order, coplanarity, and non-zero area."""
    a1 = as_point(a1)
    a2 = as_point(a2)
    b = as_point(b)
    c = as_point(c)
    leg1 = _distance(a1, b)
    leg2 = _distance(a2, c)
    if leg1 < EPS or leg2 < EPS:
        short = leg1 if leg1 < leg2 else leg2
        return ValidationResult.failure(
            f"degenerate trapezoid: leg length {short:.4f} < EPS"
        )
    diff = abs(leg1 - leg2)
    if diff >= EPS:
        return ValidationResult.failure(
            f"invalid plane — |a1b|-|a2c| = {diff:.4f} >= EPS"
        )
    base_near = _distance(a1, a2)
    base_far = _distance(b, c)
    if base_far < EPS:
        return ValidationResult.failure(
            f"degenerate trapezoid: |bc| = {base_far:.4f} < EPS"
        )
    if base_near > base_far + EPS:
        return ValidationResult.failure(
            f"invalid plane — |a1a2| = {base_near:.4f} > |bc| + EPS = {base_far + EPS:.4f}"
        )
    if not _is_coplanar(a1, a2, b, c):
        sine = _out_of_plane_sine(a1, a2, b, c)
        return ValidationResult.failure(
            f"invalid plane — points are not coplanar (|sin| = {sine:.4f})"
        )
    if _quad_area(a1, a2, b, c) < EPS:
        return ValidationResult.failure(
            "degenerate trapezoid: area is below EPS"
        )
    return ValidationResult.success()


def _out_of_plane_sine(a1, a2, b, c) -> float:
    """Absolute sine of the angle between ``a1→b`` and the plane of ``a1, a2, c``.

    Returns 0 when ``a1``, ``a2``, and ``c`` are collinear. Three collinear
    points plus any fourth point are coplanar, so a zero normal is not a failure.
    """
    normal = np.cross(a2 - a1, c - a1)
    nlen = float(np.linalg.norm(normal))
    elen = float(np.linalg.norm(b - a1))
    if nlen == 0.0 or elen == 0.0:
        return 0.0
    return abs(float(np.dot(normal, b - a1))) / (nlen * elen)


def _is_coplanar(a1, a2, b, c) -> bool:
    return _out_of_plane_sine(a1, a2, b, c) < ANGLE_TOL


def _quad_area(a1, a2, b, c) -> float:
    """Area of the quad as the sum of triangles ``a1bc`` and ``a1 c a2``."""
    area = 0.5 * float(np.linalg.norm(np.cross(b - a1, c - a1)))
    area += 0.5 * float(np.linalg.norm(np.cross(c - a1, a2 - a1)))
    return area

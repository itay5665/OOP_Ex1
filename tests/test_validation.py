"""Geometric validation rules, including the EPS boundary."""

import numpy as np

from projection.validation import EPS, validate_trapezoid, validate_triangle


def test_isosceles_triangle_passes():
    result = validate_triangle((0, 0, 0), (5, 0, 0), (0, 5, 0))
    assert result.ok


def test_equal_legs_at_eps_fail_and_half_eps_passes():
    a = (0.0, 0.0, 0.0)
    b = (5.0, 0.0, 0.0)
    just_over = (0.0, 5.0 + EPS, 0.0)
    just_under = (0.0, 5.0 + EPS / 2.0, 0.0)
    failed = validate_triangle(a, b, just_over)
    passed = validate_triangle(a, b, just_under)
    assert not failed.ok
    assert "EPS" in failed.reason
    assert passed.ok


def test_short_leg_and_short_base_fail():
    short_leg = validate_triangle((0, 0, 0), (0.0004, 0, 0), (0, 0.0004, 0))
    assert not short_leg.ok
    assert "leg" in short_leg.reason
    short_base = validate_triangle((0, 0, 0), (1, 0, 0), (1, 0.0004, 0))
    assert not short_base.ok
    assert "|bc|" in short_base.reason


def test_collinear_triangle_fails_and_spread_triangle_passes():
    collinear = validate_triangle((0, 0, 0), (2, 0, 0), (-2, 0, 0))
    assert not collinear.ok
    assert "collinear" in collinear.reason
    almost = validate_triangle((0, 0, 0), (1, 0, 0), (-1, 1e-6, 0))
    assert not almost.ok
    spread = validate_triangle((0, 0, 0), (1, 0, 0), (0, 1, 0))
    assert spread.ok


def test_rectangle_trapezoid_passes():
    result = validate_trapezoid((0, 0, 0), (0, 2, 0), (4, 0, 0), (4, 2, 0))
    assert result.ok


def test_trapezoid_leg_and_base_boundaries():
    a1, a2, b, c = _trapezoid(w_near=4.0 + EPS * 0.5, w_far=4.0)
    assert validate_trapezoid(a1, a2, b, c).ok
    a1, a2, b, c = _trapezoid(w_near=4.0 + 2.0 * EPS, w_far=4.0)
    failed = validate_trapezoid(a1, a2, b, c)
    assert not failed.ok
    assert "|a1a2|" in failed.reason

    near = validate_trapezoid((0, 0, 0), (0, 2, 0), (4, 0, 0), (4 + EPS / 2, 2, 0))
    assert near.ok
    far = validate_trapezoid((0, 0, 0), (0, 2, 0), (4, 0, 0), (4 + 2 * EPS, 2, 0))
    assert not far.ok
    assert "|a1b|-|a2c|" in far.reason


def test_non_coplanar_trapezoid_fails():
    result = validate_trapezoid((0, 0, 0), (0, 2, 0), (3, 0, 0), (0, 2, 3))
    assert not result.ok
    assert "coplanar" in result.reason


def test_zero_area_trapezoid_fails():
    result = validate_trapezoid((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
    assert not result.ok
    assert "area" in result.reason


def _trapezoid(w_near: float, w_far: float, height: float = 3.0):
    overhang = (w_far - w_near) / 2.0
    a1 = np.array([overhang, 0.0, 0.0])
    a2 = np.array([overhang + w_near, 0.0, 0.0])
    b = np.array([0.0, height, 0.0])
    c = np.array([w_far, height, 0.0])
    return a1, a2, b, c

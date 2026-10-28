"""Plane value objects.

A triangle and a trapezoid both become a ``Quad`` before any sampling happens.
The projector never inspects the original point count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def as_point(coords) -> np.ndarray:
    """Return a finite ``float64`` point of shape ``(3,)``.

    A copy is returned so later code cannot mutate a caller's array through
    the value object.
    """
    arr = np.asarray(coords, dtype=np.float64).reshape(3).copy()
    if not np.all(np.isfinite(arr)):
        raise ValueError("coordinates must be finite")
    return arr


@dataclass(frozen=True)
class Quad:
    """Four corners of the shape that is sampled.

    ``start1 → end1`` and ``start2 → end2`` are the two equal legs.
    ``end1 → end2`` is the far base.
    """

    start1: np.ndarray
    end1: np.ndarray
    start2: np.ndarray
    end2: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "start1", as_point(self.start1))
        object.__setattr__(self, "end1", as_point(self.end1))
        object.__setattr__(self, "start2", as_point(self.start2))
        object.__setattr__(self, "end2", as_point(self.end2))


class PlaneSpec(ABC):
    """A validated plane from the CSV. Subclasses only know how to become a quad."""

    @abstractmethod
    def to_quad(self) -> Quad:
        """Return the corners the projector samples."""


@dataclass(frozen=True)
class TrianglePlane(PlaneSpec):
    """Isosceles triangle ``(a, b, c)`` with equal legs ``a→b`` and ``a→c``."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", as_point(self.a))
        object.__setattr__(self, "b", as_point(self.b))
        object.__setattr__(self, "c", as_point(self.c))

    def to_quad(self) -> Quad:
        return Quad(self.a, self.b, self.a, self.c)


@dataclass(frozen=True)
class TrapezoidPlane(PlaneSpec):
    """Isosceles trapezoid. The legs are ``a1→b`` and ``a2→c``."""

    a1: np.ndarray
    a2: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "a1", as_point(self.a1))
        object.__setattr__(self, "a2", as_point(self.a2))
        object.__setattr__(self, "b", as_point(self.b))
        object.__setattr__(self, "c", as_point(self.c))

    def to_quad(self) -> Quad:
        return Quad(self.a1, self.b, self.a2, self.c)

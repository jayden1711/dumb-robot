"""Courses: terrain geoms, start pose, finish rule, segments. Terrain uses contype/conaffinity 2."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco

from .bodylib import TERRAIN_BIT

GEOM = mujoco.mjtGeom


@dataclass
class Segment:
    name: str
    x0: float
    x1: float


@dataclass
class Course:
    name: str
    start_x: float
    finish_x: float
    t_limit: float
    e_ref: float  # J, energy scale for the scoring tie-break
    segments: list[Segment]
    terrain_mu: float = 0.8
    builder: callable = field(default=None, repr=False)

    @property
    def length(self) -> float:
        return self.finish_x - self.start_x

    def progress(self, x: float, y: float = 0.0) -> float:
        return x - self.start_x

    def segment_index(self, x: float) -> int:
        for i, s in enumerate(self.segments):
            if x < s.x1:
                return i
        return len(self.segments) - 1

    def add_to(self, spec) -> list[tuple[str, float]]:
        """Add terrain geoms to the spec; return [(geom name, terrain friction)]."""
        return self.builder(spec, self)


def _terrain_geom(spec, geoms, mu, name, gtype, size, pos, **kw):
    spec.worldbody.add_geom(name=name, type=gtype, size=list(size) + [0] * (3 - len(size)), pos=list(pos),
                            contype=TERRAIN_BIT, conaffinity=TERRAIN_BIT, rgba=kw.pop("rgba", [0.5, 0.5, 0.55, 1]), **kw)
    geoms.append((name, mu))


def _floor(spec, geoms, mu):
    _terrain_geom(spec, geoms, mu, "floor", GEOM.mjGEOM_PLANE, [20, 20, 0.1], (0, 0, 0))


BUMP_HEIGHTS = (0.008, 0.016, 0.024, 0.032, 0.040)
BUMP_RADIUS = 0.12
BUMP_X0, BUMP_DX = 1.0, 0.7
LANE_HALF = 0.45


def _bumps(spec, course):
    geoms: list = []
    mu = course.terrain_mu
    _floor(spec, geoms, mu)
    L = course.finish_x + 0.4
    for sy in (+1, -1):
        _terrain_geom(spec, geoms, mu, f"wall_{'l' if sy > 0 else 'r'}", GEOM.mjGEOM_BOX, [L / 2, 0.01, 0.1],
                      (L / 2 - 0.2, sy * (LANE_HALF + 0.01), 0.1))
    for i, h in enumerate(BUMP_HEIGHTS):
        _terrain_geom(spec, geoms, mu, f"bump_{i + 1}", GEOM.mjGEOM_CYLINDER, [BUMP_RADIUS, LANE_HALF],
                      (BUMP_X0 + i * BUMP_DX, 0, h - BUMP_RADIUS), euler=[math.pi / 2, 0, 0],
                      rgba=[0.7, 0.4, 0.4, 1])
    return geoms


def _flat(spec, course):
    geoms: list = []
    _floor(spec, geoms, course.terrain_mu)
    return geoms


def bumps_course() -> Course:
    segs = [Segment("approach", -1e9, BUMP_X0 - 0.35)]
    for i in range(len(BUMP_HEIGHTS)):
        c = BUMP_X0 + i * BUMP_DX
        segs.append(Segment(f"bump_{i + 1}", c - 0.35, c + 0.35))
    finish = BUMP_X0 + (len(BUMP_HEIGHTS) - 1) * BUMP_DX + 1.0
    segs.append(Segment("run_out", segs[-1].x1, finish))
    return Course("bumps", 0.25, finish, 25.0, 25.0, segs, builder=_bumps)


def flat_course() -> Course:
    return Course("flat", 0.0, 3.0, 5.0, 25.0, [Segment("flat", -1e9, 3.0)], builder=_flat)


COURSES = {"bumps": bumps_course, "flat": flat_course}


def get_course(name: str) -> Course:
    return COURSES[name]()

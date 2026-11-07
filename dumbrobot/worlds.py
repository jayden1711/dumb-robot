"""Courses: terrain geoms, start pose, finish rule, segments. Terrain uses contype/conaffinity 2."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

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
    path: list | None = None  # centre-line waypoints [(x, y)]; progress is arc length along it (start_x + length so far)

    def __post_init__(self):
        self._ps = None
        if self.path:
            pts = [np.asarray(p, dtype=float) for p in self.path]
            xs, ys, ss, s = [], [], [], self.start_x
            for a, b in zip(pts[:-1], pts[1:]):
                seg = float(np.linalg.norm(b - a))
                n = max(1, int(math.ceil(seg / 0.01)))
                for k in range(n):
                    p = a + (b - a) * (k / n)
                    xs.append(p[0]), ys.append(p[1]), ss.append(s + seg * k / n)
                s += seg
            xs.append(pts[-1][0]), ys.append(pts[-1][1]), ss.append(s)
            self._px, self._py, self._ps = np.array(xs), np.array(ys), np.array(ss)

    def waypoint_s(self) -> list[float]:
        """Course coordinate s of each waypoint."""
        out, s = [self.start_x], self.start_x
        for a, b in zip(self.path[:-1], self.path[1:]):
            s += math.hypot(b[0] - a[0], b[1] - a[1])
            out.append(s)
        return out

    @property
    def length(self) -> float:
        return self.finish_x - self.start_x

    def s_of(self, x: float, y: float, hint: float | None = None) -> float:
        """Course coordinate of a point: x itself on straight courses; on a path course the nearest centre-line sample
        within a window around `hint` (the furthest s so far), so a robot cannot be credited across a wall."""
        if self._ps is None:
            return x
        i0 = 0 if hint is None else int(np.searchsorted(self._ps, hint))
        lo, hi = max(0, i0 - 50), min(len(self._ps), i0 + 80)
        j = lo + int(np.argmin((self._px[lo:hi] - x) ** 2 + (self._py[lo:hi] - y) ** 2))
        return float(self._ps[j])

    def progress(self, s: float, y: float = 0.0) -> float:
        return s - self.start_x

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


# ---- course 2: corridor with a perpendicular wall, then a two-baffle chicane -------------------------------------------
CORR_HALF = 0.55  # inner half-width of the corridor
CORR_END = 6.2
PILLAR_X = 1.5  # perpendicular wall across the corridor from the right wall to PILLAR_Y1, gap on the left
PILLAR_Y1 = 0.05
BAFFLES = (  # slanted guide walls: (x_start, y_start, x_end, y_end); the second leans the other way
    ((3.0, CORR_HALF), (3.6, -0.15)),
    ((4.3, -CORR_HALF), (4.9, 0.15)),
)
CORR_PATH = [(0.25, 0.0), (1.1, 0.0), (1.5, 0.3), (1.95, 0.3), (2.9, 0.0), (3.6, -0.38), (4.2, -0.38), (4.9, 0.38),
             (5.6, 0.0)]
WALL_THICK, WALL_H = 0.01, 0.2  # half thickness, half height


def _corridor(spec, course):
    geoms: list = []
    mu = course.terrain_mu
    _floor(spec, geoms, mu)
    L = CORR_END + 0.4
    for sy in (+1, -1):
        _terrain_geom(spec, geoms, mu, f"wall_{'l' if sy > 0 else 'r'}", GEOM.mjGEOM_BOX, [L / 2, WALL_THICK, WALL_H],
                      (L / 2 - 0.2, sy * (CORR_HALF + WALL_THICK), WALL_H))
    _terrain_geom(spec, geoms, mu, "pillar", GEOM.mjGEOM_BOX, [WALL_THICK, (PILLAR_Y1 + CORR_HALF) / 2, WALL_H],
                  (PILLAR_X, (PILLAR_Y1 - CORR_HALF) / 2, WALL_H), rgba=[0.7, 0.4, 0.4, 1])
    for i, ((x0, y0), (x1, y1)) in enumerate(BAFFLES):
        length = math.hypot(x1 - x0, y1 - y0)
        _terrain_geom(spec, geoms, mu, f"baffle_{i + 1}", GEOM.mjGEOM_BOX, [length / 2, WALL_THICK, WALL_H],
                      ((x0 + x1) / 2, (y0 + y1) / 2, WALL_H), euler=[0, 0, math.atan2(y1 - y0, x1 - x0)],
                      rgba=[0.7, 0.4, 0.4, 1])
    return geoms


def corridor_course() -> Course:
    c = Course("corridor", CORR_PATH[0][0], 0.0, 25.0, 25.0, [], builder=_corridor, path=list(CORR_PATH))
    S = c.waypoint_s()
    c.finish_x = S[-1]
    c.segments = [Segment("approach", -1e9, S[1]), Segment("wall", S[1], S[3]), Segment("corridor", S[3], S[4]),
                  Segment("chicane_1", S[4], S[6]), Segment("chicane_2", S[6], S[7]), Segment("run_out", S[7], S[-1])]
    return c


COURSES = {"bumps": bumps_course, "corridor": corridor_course, "flat": flat_course}


def get_course(name: str) -> Course:
    return COURSES[name]()

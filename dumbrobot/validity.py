"""Validity gate: run before any scoring. Any failure rejects the proposal (PLAN section 4)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import schema
from .assembly import BuildError, assemble
from .bodylib import MIN_BODY_MASS, ROBOT_BIT
from .brains import BrainA
from .sim import run_episode
from .worlds import flat_course

MASS_RANGE = (0.15, 3.0)
ENVELOPE = (0.45, 0.40, 0.25)  # max x, y, z extent at the rest pose [m]
PENETRATION_TOL = 0.0015
GROUND_TOL = 0.002
SETTLE_S = 1.0
SETTLE_SPEED_MAX = 5.0
SETTLE_UP_MIN = 0.9
DRIVE_S = 1.0
DRIVE_MIN_DX = 0.01


@dataclass
class Report:
    ok: bool
    failures: list[dict] = field(default_factory=list)

    def add(self, name, detail):
        self.failures.append(dict(name=name, detail=detail))
        self.ok = False

    def text(self) -> str:
        return "ok" if self.ok else "; ".join(f"{f['name']}: {f['detail']}" for f in self.failures)


def _extent(m, d, geom_ids):
    pts = []
    for g in geom_ids:
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(d.geom_xpos[g] + R @ (c + h * np.array([sx, sy, sz])))
    pts = np.array(pts)
    return pts.max(axis=0) - pts.min(axis=0)


def check_design(build_fn, genome) -> Report:
    rep = Report(True)
    errs = schema.validate(genome)
    if errs:
        rep.add("schema", "; ".join(errs))
        return rep
    flat = flat_course()
    try:
        asm = assemble(build_fn, genome, flat, spawn_dz=0.0, self_pairs=True)
    except BuildError as e:
        rep.add("build", str(e))
        return rep
    m = asm.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    total = float(np.sum(m.body_mass))
    if not MASS_RANGE[0] <= total <= MASS_RANGE[1]:
        rep.add("mass", f"total {total:.3f} kg outside {MASS_RANGE}")
    light = [m.body(i).name for i in range(1, m.nbody) if m.body_mass[i] < MIN_BODY_MASS]
    if light:
        rep.add("mass", f"bodies below {MIN_BODY_MASS} kg: {light}")

    robot_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] == ROBOT_BIT]
    non_terrain = [g for g in range(m.ngeom) if m.geom_contype[g] != 2]
    ext = _extent(m, d, non_terrain)
    if any(e > lim for e, lim in zip(ext, ENVELOPE)):
        rep.add("envelope", f"extent {np.round(ext, 3).tolist()} m exceeds {list(ENVELOPE)}")

    is_robot = {g: m.geom_contype[g] == ROBOT_BIT for g in range(m.ngeom)}
    bad = sorted({f"{m.geom(c.geom1).name}~{m.geom(c.geom2).name} ({-c.dist * 1000:.1f} mm)"
                  for c in d.contact[: d.ncon]
                  if is_robot[c.geom1] and is_robot[c.geom2] and c.dist < -PENETRATION_TOL})
    if bad:
        rep.add("intersect", "penetrating parts at rest pose: " + ", ".join(bad))

    clip = sorted({f"{m.geom(c.geom1).name}~{m.geom(c.geom2).name} ({-c.dist * 1000:.1f} mm)"
                   for c in d.contact[: d.ncon]
                   if not (is_robot[c.geom1] and is_robot[c.geom2]) and c.dist < -GROUND_TOL})
    if clip:
        rep.add("ground_clip", "parts below the ground plane at rest pose: " + ", ".join(clip))
    if not rep.ok:
        return rep

    try:
        asm = assemble(build_fn, genome, flat, spawn_dz=0.01)
    except BuildError as e:
        rep.add("build", str(e))
        return rep
    m = asm.model
    d = mujoco.MjData(m)
    bid = m.body("chassis").id
    for _ in range(int(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or float(np.linalg.norm(d.qvel[:3])) > SETTLE_SPEED_MAX:
            rep.add("explode", "non-finite state or speed above limit while settling")
            return rep
    if float(d.xmat[bid][8]) < SETTLE_UP_MIN:
        rep.add("unstable", f"does not rest upright (up-axis {float(d.xmat[bid][8]):.2f})")
        return rep

    drive_course = dataclasses.replace(flat, t_limit=DRIVE_S)
    asm = assemble(build_fn, genome, drive_course)
    res = run_episode(asm, BrainA())
    if res.status == "exploded":
        rep.add("explode", "exploded during the flat-ground drive test")
    elif res.flips:
        rep.add("drive", "flips on flat ground under Brain A")
    elif res.progress * drive_course.length < DRIVE_MIN_DX:
        rep.add("drive", f"moves less than {DRIVE_MIN_DX * 100:.0f} cm in {DRIVE_S:.0f} s on flat ground under Brain A")
    return rep

"""Episode runner: drives the model with a frozen brain, logs per-segment stats and an end-of-episode contact snapshot."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import catalog as C

UP_FLIP = 0.2
UP_RECOVER = 0.5
STUCK_WINDOW_S = 2.0
STUCK_MIN_GAIN = 0.01
STUCK_END_S = 4.0
FLIPPED_END_S = 3.0
SPEED_EXPLODE = 20.0
TRACE_DT = 0.05
SLIP_SURFACE_MIN = 0.1  # m/s
SLIP_RATIO = 0.3


def motor_power(tau: float, w: float, stall: float, w0: float) -> float:
    """Electrical power of a DC motor, no regeneration credit: shaft power plus resistive loss tau^2 w0 / stall."""
    return max(0.0, tau * w) + tau * tau * w0 / stall


class FlipCounter:
    """A flip is up-axis z dropping below UP_FLIP after having been above UP_RECOVER."""

    def __init__(self):
        self.upright = True
        self.flips = 0
        self.inverted_s = 0.0

    def update(self, up_z: float, dt: float) -> bool:
        event = False
        if self.upright and up_z < UP_FLIP:
            self.upright = False
            self.flips += 1
            event = True
        elif not self.upright and up_z > UP_RECOVER:
            self.upright = True
        self.inverted_s = 0.0 if self.upright else self.inverted_s + dt
        return event


@dataclass
class SegStat:
    name: str
    x0: float
    x1: float
    entered_t: float | None = None
    left_t: float | None = None
    ticks: int = 0
    stuck_s: float = 0.0
    slip_s: float = 0.0
    flips: int = 0
    energy_j: float = 0.0
    min_up: float = 1.0
    status: str = "not_reached"
    progress_m: float = 0.0
    duration_s: float = 0.0
    mean_speed: float = 0.0


@dataclass
class EpisodeResult:
    status: str  # finished | timeout | stuck | flipped_stuck | exploded
    done: bool
    progress: float  # 0..1 of course length
    time_s: float
    energy_j: float
    flips: int
    segments: list[SegStat]
    ending: dict
    brain: str
    course: str
    trace: dict | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("status", "done", "progress", "time_s", "energy_j", "flips", "ending", "brain",
                                           "course")}
        d["segments"] = [s.__dict__ for s in self.segments]
        return d


def _name(m, geom_id):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id)


TOUCH_DIST = 0.003


def _touching(m, d):
    """Robot-terrain pairs within TOUCH_DIST (contacts flicker at zero margin, so measure distance instead)."""
    out = []
    for p in range(m.npair):
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        if mujoco.mj_geomDistance(m, d, g1, g2, TOUCH_DIST, None) < TOUCH_DIST:
            out.append([_name(m, g1), _name(m, g2)])
    return out


def run_episode(asm, brain, record_trace: bool = False) -> EpisodeResult:
    m, course, b = asm.model, asm.course, asm.builder
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    brain.reset()
    bid = m.body("chassis").id
    nsub = max(1, round(C.CONTROL_DT / m.opt.timestep))
    dt = nsub * m.opt.timestep
    act_ids = [m.actuator(n).id for n in b.actuators]
    wj = [m.joint(n).dofadr[0] for n in b.wheel_joints]
    wq = [m.joint(n).qposadr[0] for n in b.wheel_joints]
    bump_adr = m.joint(b.bumper_joint).qposadr[0] if b.bumper_joint else None
    stall, w0, r = b.motor["stall"], b.motor["w0"], b.wheel_r

    segs = [SegStat(s.name, s.x0, s.x1) for s in course.segments]
    flipc = FlipCounter()
    x_hist: list[float] = []
    xmax = float(d.xpos[bid][0])
    energy = 0.0
    stuck_clock = 0.0
    status = None
    trace_t, trace_p, trace_q, trace_w = [], [], [], []
    next_trace = 0.0
    window = int(round(STUCK_WINDOW_S / dt))
    last_seg = 0

    while status is None:
        sensors = {"bumper": bool(bump_adr is not None and d.qpos[bump_adr] < -b.bumper_threshold)}
        u = brain.act(d.time, sensors)
        d.ctrl[: len(u)] = u
        mujoco.mj_step(m, d, nstep=nsub)
        t = d.time
        pos = d.xpos[bid]
        if not np.all(np.isfinite(d.qpos)) or float(np.max(np.abs(d.qvel))) > SPEED_EXPLODE * 50:
            status = "exploded"
            break
        vel = d.qvel[:3]  # chassis is the root freejoint: world-frame linear velocity
        if float(np.linalg.norm(vel)) > SPEED_EXPLODE:
            status = "exploded"
            break
        up_z = float(d.xmat[bid][8])
        heading = d.xmat[bid][[0, 3]]
        fwd_speed = float(vel[0] * heading[0] + vel[1] * heading[1])
        xmax = max(xmax, float(pos[0]))
        x_hist.append(xmax)

        power = sum(motor_power(float(d.actuator_force[i]), float(d.qvel[j]), stall, w0) for i, j in zip(act_ids, wj))
        energy += power * dt
        flip_event = flipc.update(up_z, dt)

        gained = len(x_hist) > window and (xmax - x_hist[-1 - window]) < STUCK_MIN_GAIN
        stuck_clock = stuck_clock + dt if gained else 0.0
        seg_i = course.segment_index(xmax)
        s = segs[seg_i]
        if s.entered_t is None:
            s.entered_t = t - dt
            for k in range(last_seg, seg_i):
                if segs[k].entered_t is not None:
                    segs[k].left_t = t - dt
        last_seg = seg_i
        s.ticks += 1
        s.energy_j += power * dt
        s.min_up = min(s.min_up, up_z)
        s.flips += int(flip_event)
        if gained:
            s.stuck_s += dt
        surface = float(np.mean([abs(d.qvel[j]) for j in wj])) * r
        if surface > SLIP_SURFACE_MIN and fwd_speed < SLIP_RATIO * surface:
            s.slip_s += dt

        if record_trace and t >= next_trace:
            trace_t.append(t)
            trace_p.append(pos.copy())
            trace_q.append(d.xquat[bid].copy())
            trace_w.append([d.qpos[q] for q in wq])
            next_trace += TRACE_DT

        if xmax >= course.finish_x and up_z > UP_FLIP:
            status = "finished"
        elif t >= course.t_limit:
            status = "timeout"
        elif stuck_clock >= STUCK_END_S:
            status = "stuck"
        elif flipc.inverted_s >= FLIPPED_END_S:
            status = "flipped_stuck"

    done = status == "finished"
    t_end = float(d.time)
    for k, s in enumerate(segs):
        if s.entered_t is None:
            continue
        crossed = xmax >= s.x1 or (done and k == len(segs) - 1)
        s.status = "crossed" if crossed else "failed_here"
        end_t = s.left_t if s.left_t is not None else t_end
        s.duration_s = max(0.0, end_t - s.entered_t)
        lo = max(s.x0, course.start_x)
        s.progress_m = max(0.0, min(xmax, s.x1) - lo)
        s.mean_speed = s.progress_m / s.duration_s if s.duration_s > 0 else 0.0
    ending = {"x": float(d.xpos[bid][0]), "up_z": float(d.xmat[bid][8]) if status != "exploded" else float("nan")}
    if status in ("stuck", "flipped_stuck", "timeout"):
        ending["contacts"] = _touching(m, d)
    trace = None
    if record_trace and trace_t:
        trace = dict(t=np.array(trace_t), pos=np.array(trace_p), quat=np.array(trace_q), wheel=np.array(trace_w))
    progress = 0.0 if status == "exploded" else float(np.clip(course.progress(xmax) / course.length, 0.0, 1.0))
    return EpisodeResult(status, done, progress, t_end, float(energy), flipc.flips, segs, ending, brain.name, course.name,
                         trace)


def summarize(res: EpisodeResult) -> str:
    """Plain-text failure log for a designer: facts only, no advice."""
    lines = [f"course={res.course} brain={res.brain} outcome={res.status} progress={res.progress:.2f} t={res.time_s:.1f}s "
             f"energy={res.energy_j:.1f}J flips={res.flips}"]
    for s in res.segments:
        if s.status == "not_reached":
            lines.append(f"  {s.name}: not reached")
            continue
        lines.append(f"  {s.name}: {s.status}, {s.duration_s:.1f}s inside, {s.progress_m:.2f}m gained, "
                     f"mean {s.mean_speed:.2f}m/s, stuck {s.stuck_s:.1f}s, wheel-slip {s.slip_s:.1f}s, flips {s.flips}, "
                     f"min up-axis {s.min_up:.2f}, {s.energy_j:.1f}J")
    if not res.done:
        e = res.ending
        c = e.get("contacts")
        lines.append(f"  ended at x={e['x']:.2f}m, up-axis={e['up_z']:.2f}" +
                     (f", touching: {', '.join(f'{a}-{b}' for a, b in c)}" if c else ""))
    return "\n".join(lines)

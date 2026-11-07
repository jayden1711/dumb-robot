"""CPU-only clip renderer: numpy software rasterizer + a minimal GIF89a writer (no GL, no ffmpeg, single-threaded).

Each frame has a top-down panel (x-y) and a side panel (x-z), both following the chassis. Robot geoms are taken from the
assembled model at the rest pose and moved rigidly with the chassis pose from the trace (joint motion is not shown).
Simulation only.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .assembly import assemble, load_script
from .bodylib import ROBOT_BIT, TERRAIN_BIT
from .worlds import get_course

GT = mujoco.mjtGeom
MAX_FRAMES = 250
WINDOW_M = 2.4  # horizontal extent of both panels
TOP_HALF_M = 0.6  # top panel shows y in [-0.6, 0.6] around the chassis
SIDE_ZMAX, SIDE_ZMIN = 0.35, -0.05
DIVIDER = 4

PALETTE = np.array([
    (235, 235, 228),  # 0 background
    (50, 50, 60),     # 1 terrain (top view)
    (230, 140, 30),   # 2 chassis
    (20, 20, 20),     # 3 wheel
    (210, 50, 50),    # 4 bumper plates
    (150, 150, 150),  # 5 caster
    (90, 90, 200),    # 6 other robot parts (whiskers, tail, ...)
    (40, 170, 70),    # 7 finish line
    (180, 100, 100),  # 8 bump cylinders / pillar
    (150, 150, 160),  # 9 terrain (side view)
    (170, 150, 120),  # 10 ground
    (0, 0, 0),        # 11 divider
] + [(255, 255, 255)] * 4, dtype=np.uint8)


# ---- GIF writer ----------------------------------------------------------------------------------------------------
def _lzw(data: bytes, min_code: int) -> bytes:
    clear, eoi = 1 << min_code, (1 << min_code) + 1
    out, acc, nacc = bytearray(), 0, 0
    nxt, size, table = eoi + 1, min_code + 1, {}

    def emit(code, size):
        nonlocal acc, nacc
        acc |= code << nacc
        nacc += size
        while nacc >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nacc -= 8

    emit(clear, size)
    prefix = data[0]
    for b in data[1:]:
        key = (prefix << 8) | b
        c = table.get(key)
        if c is not None:
            prefix = c
            continue
        emit(prefix, size)
        table[key] = nxt
        if nxt == (1 << size) and size < 12:
            size += 1
        nxt += 1
        if nxt > 4095:
            emit(clear, size)
            table.clear()
            nxt, size = eoi + 1, min_code + 1
        prefix = b
    emit(prefix, size)
    emit(eoi, size)
    if nacc:
        out.append(acc & 0xFF)
    return bytes(out)


def write_gif(frames, palette, path, fps: float = 10.0) -> str:
    """frames: list of equal-shape 2-D uint8 index arrays; palette: (n, 3) uint8. Loops forever."""
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    h, w = frames[0].shape
    pal = np.asarray(palette, dtype=np.uint8)
    bits = max(2, int(math.ceil(math.log2(max(2, len(pal))))))
    table = np.zeros((1 << bits, 3), dtype=np.uint8)
    table[: len(pal)] = pal
    delay = max(2, int(round(100.0 / fps)))
    buf = bytearray(b"GIF89a")
    buf += w.to_bytes(2, "little") + h.to_bytes(2, "little") + bytes([0x80 | ((bits - 1) << 4) | (bits - 1), 0, 0])
    buf += table.tobytes()
    buf += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
    for f in frames:
        buf += b"\x21\xf9\x04\x00" + delay.to_bytes(2, "little") + b"\x00\x00"
        buf += b"\x2c" + (0).to_bytes(2, "little") * 2 + w.to_bytes(2, "little") + h.to_bytes(2, "little") + b"\x00"
        buf.append(bits)
        data = _lzw(f.tobytes(), bits)
        for i in range(0, len(data), 255):
            blk = data[i:i + 255]
            buf.append(len(blk))
            buf += blk
        buf.append(0)
    buf.append(0x3B)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(buf))
    return str(p)


# ---- geometry ------------------------------------------------------------------------------------------------------
def _shape_points(gtype, size):
    """Points (n, 3) in the geom frame whose projected convex hull covers the shape, plus a radius to grow each point by."""
    if gtype == GT.mjGEOM_BOX:
        sx, sy, sz = size[:3]
        return np.array([(a * sx, b * sy, c * sz) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]), 0.0
    if gtype == GT.mjGEOM_CYLINDER:
        r, hl = size[0], size[1]
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        return np.array([(r * math.cos(a), r * math.sin(a), s * hl) for a in ang for s in (-1, 1)]), 0.0
    if gtype == GT.mjGEOM_SPHERE:
        return np.zeros((1, 3)), float(size[0])
    if gtype == GT.mjGEOM_CAPSULE:
        return np.array([(0, 0, -size[1]), (0, 0, size[1])], dtype=float), float(size[0])
    return None, 0.0


def _hull(P):
    pts = sorted(set(map(tuple, np.round(P, 5).tolist())))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h

    lo, up = half(pts), half(pts[::-1])
    return np.array(lo[:-1] + up[:-1])


_CIRC = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)])


def _hull2(world_pts, rad, ax):
    P = world_pts[:, ax]
    if rad > 0:
        P = (P[:, None, :] + rad * _CIRC[None]).reshape(-1, 2)
    return _hull(P)


def _fill(img, poly, color, ymin, ymax):
    if len(poly) == 0:
        return
    H, W = img.shape
    x0, x1 = max(0, int(math.floor(poly[:, 0].min()))), min(W, int(math.ceil(poly[:, 0].max())) + 1)
    y0, y1 = max(ymin, int(math.floor(poly[:, 1].min()))), min(ymax, int(math.ceil(poly[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    if len(poly) < 3 or abs(area) < 1.0:
        img[y0:y1, x0:x1] = color
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px, py = xx + 0.5, yy + 0.5
    inside = np.ones(px.shape, dtype=bool)
    sgn = 1.0 if area > 0 else -1.0
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        inside &= sgn * ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) >= -0.5 * np.hypot(*(b - a))
    img[y0:y1, x0:x1][inside] = color


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _robot_color(name: str) -> int:
    if name == "chassis":
        return 2
    if name.startswith("wheel"):
        return 3
    if name.startswith("bump"):
        return 4
    if name.startswith("caster"):
        return 5
    return 6


def _collect(asm):
    m = asm.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = m.body("chassis").id
    Rc, pc = d.xmat[bid].reshape(3, 3), d.xpos[bid].copy()
    robot, terrain = [], []
    for g in range(m.ngeom):
        pts, rad = _shape_points(m.geom_type[g], m.geom_size[g])
        if pts is None:
            continue
        name = m.geom(g).name or ""
        Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_contype[g] == ROBOT_BIT:
            local = (Rc.T @ ((Rg @ pts.T).T + pg - pc).T).T
            robot.append((_robot_color(name), local, rad))
        elif m.geom_contype[g] == TERRAIN_BIT:
            world = ((Rg @ pts.T).T + pg)
            terrain.append((name, world, rad))
    robot.sort(key=lambda r: r[0] != 2)  # chassis first, then the rest on top
    return robot, terrain


def render_clip(body_py_path, course_name, trace, out_path, fps=10, width=480) -> str:
    genome, build_fn = load_script(body_py_path)
    course = get_course(course_name)
    robot, terrain = _collect(assemble(build_fn, genome, course))
    scale = width / WINDOW_M
    top_h = int(round(2 * TOP_HALF_M * scale))
    side_h = int(round((SIDE_ZMAX - SIDE_ZMIN) * scale))
    H, W = top_h + DIVIDER + side_h, width
    side0 = top_h + DIVIDER
    ground_row = side0 + int(round((SIDE_ZMAX - 0.0) * scale))

    # terrain hulls in world coordinates, projected once per view
    t_top = [((8 if n.startswith("bump") or n == "pillar" else 1), _hull2(w, r, [0, 1])) for n, w, r in terrain]
    t_side = [(8 if n.startswith("bump") else 9, _hull2(w, r, [0, 2])) for n, w, r in terrain]
    finish_x = course.path[-1][0] if course.path else course.finish_x

    t, pos, quat = np.asarray(trace["t"]), np.asarray(trace["pos"]), np.asarray(trace["quat"])
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 0.05
    step = max(1, int(round(1.0 / (fps * dt))))
    step = max(step, int(math.ceil(n / MAX_FRAMES)))
    frames = []
    for i in range(0, n, step):
        cx, cy = float(pos[i][0]), float(pos[i][1])
        R, p = _quat_mat(quat[i]), pos[i]
        img = np.zeros((H, W), dtype=np.uint8)
        img[side0 - DIVIDER:side0] = 11
        img[ground_row:] = 10

        def sx(x):
            return (x - cx) * scale + W / 2

        for col, hull in t_top:
            if len(hull):
                _fill(img, np.stack([sx(hull[:, 0]), top_h / 2 - (hull[:, 1] - cy) * scale], 1), col, 0, top_h)
        for col, hull in t_side:
            if len(hull):
                _fill(img, np.stack([sx(hull[:, 0]), side0 + (SIDE_ZMAX - hull[:, 1]) * scale], 1), col, side0, H)
        fx = int(round(sx(finish_x)))
        if 0 <= fx < W:
            img[0:top_h, fx] = 7
            img[side0:H, fx] = 7
        for col, local, rad in robot:
            w = (R @ local.T).T + p
            h = _hull2(w, rad, [0, 1])
            if len(h):
                _fill(img, np.stack([sx(h[:, 0]), top_h / 2 - (h[:, 1] - cy) * scale], 1), col, 0, top_h)
            h = _hull2(w, rad, [0, 2])
            if len(h):
                _fill(img, np.stack([sx(h[:, 0]), side0 + (SIDE_ZMAX - h[:, 1]) * scale], 1), col, side0, H)
        frames.append(img)
    return write_gif(frames, PALETTE, out_path, fps=1.0 / (step * dt))

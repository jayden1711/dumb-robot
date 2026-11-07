from pathlib import Path

import numpy as np
import pytest

from dumbrobot.render import PALETTE, render_clip, write_gif

ROOT = Path(__file__).resolve().parents[1]


def _lzw_decode(data: bytes, min_code: int):
    clear, eoi = 1 << min_code, (1 << min_code) + 1
    bitpos, out = 0, []
    size = min_code + 1
    table = None
    prev = None
    total = len(data) * 8

    def reset():
        return {i: bytes([i]) for i in range(clear)} | {clear: b"", eoi: b""}

    while bitpos + size <= total:
        code = 0
        for k in range(size):
            code |= ((data[(bitpos + k) >> 3] >> ((bitpos + k) & 7)) & 1) << k
        bitpos += size
        if code == clear:
            table, size, prev = reset(), min_code + 1, None
            continue
        if code == eoi:
            break
        if prev is None:
            entry = table[code]
        else:
            entry = table[code] if code in table else table[prev] + table[prev][:1]
            table[len(table)] = table[prev] + entry[:1]
            if len(table) == (1 << size) and size < 12:
                size += 1
        out.append(entry)
        prev = code
    return b"".join(out)


def parse_gif(buf: bytes):
    assert buf[:6] == b"GIF89a" and buf[-1] == 0x3B
    w, h = int.from_bytes(buf[6:8], "little"), int.from_bytes(buf[8:10], "little")
    packed = buf[10]
    i = 13 + (3 * (1 << ((packed & 7) + 1)) if packed & 0x80 else 0)
    frames = []
    while buf[i] != 0x3B:
        if buf[i] == 0x21:
            i += 2
            while buf[i]:
                i += buf[i] + 1
            i += 1
        elif buf[i] == 0x2C:
            i += 10
            mc = buf[i]
            i += 1
            data = bytearray()
            while buf[i]:
                data += buf[i + 1:i + 1 + buf[i]]
                i += buf[i] + 1
            i += 1
            frames.append(np.frombuffer(_lzw_decode(bytes(data), mc), dtype=np.uint8).reshape(h, w))
        else:
            raise AssertionError(f"bad block {buf[i]:#x}")
    return frames


def test_gif_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    f1 = rng.integers(0, 12, size=(40, 60), dtype=np.uint8)
    f2 = np.zeros((40, 60), dtype=np.uint8)
    f2[10:30, 5:50] = 3
    p = write_gif([f1, f2], PALETTE, tmp_path / "t.gif", fps=10)
    frames = parse_gif(Path(p).read_bytes())
    assert len(frames) == 2
    assert np.array_equal(frames[0], f1) and np.array_equal(frames[1], f2)


def test_gif_roundtrip_forces_table_reset(tmp_path):
    rng = np.random.default_rng(1)
    f = rng.integers(0, 12, size=(200, 300), dtype=np.uint8)  # noise: fills the 4096-entry table several times
    p = write_gif([f], PALETTE, tmp_path / "n.gif")
    assert np.array_equal(parse_gif(Path(p).read_bytes())[0], f)


def _trace(x0, x1, n=60):
    t = np.arange(n) * 0.05
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(x0, x1, n)
    pos[:, 2] = 0.06
    quat = np.tile([1.0, 0, 0, 0], (n, 1))
    return dict(t=t, pos=pos, quat=quat, wheel=np.zeros((n, 2)))


@pytest.mark.parametrize("course,x0,x1", [("bumps", 0.25, 1.6), ("corridor", 0.25, 1.4)])
def test_render_synthetic_trace(tmp_path, course, x0, x1):
    out = render_clip(ROOT / "dumbrobot" / "body_template.py", course, _trace(x0, x1), tmp_path / f"{course}.gif")
    buf = Path(out).read_bytes()
    assert buf[:6] == b"GIF89a"
    frames = parse_gif(buf)
    assert len(frames) > 5
    assert len({f.tobytes() for f in frames}) > 1  # the view moves
    assert {2, 3} <= set(np.unique(frames[0]).tolist())  # chassis and wheels are drawn

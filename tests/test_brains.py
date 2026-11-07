from pathlib import Path

from dumbrobot.brains import BrainA, BrainB

ROOT = Path(__file__).resolve().parents[1]


def test_brain_a_constant_and_stateless():
    a = BrainA()
    outs = {a.act(t, s) for t in (0.0, 1.3, 99.0) for s in ({}, {"bumper": True}, {"bumper": False, "x": 1})}
    assert outs == {(1.0, 1.0)}
    assert vars(a) == {}


def test_brain_b_reflex_lasts_half_a_second():
    b = BrainB()
    assert b.act(0.0, {"bumper": False}) == (1.0, 1.0)
    first = b.act(1.0, {"bumper": True})
    assert first[0] < 0 and first[1] < 0
    assert b.act(1.4, {"bumper": False}) == first
    assert b.act(1.51, {"bumper": False}) == (1.0, 1.0)


def test_brains_do_not_import_design_code():
    src = (ROOT / "dumbrobot" / "brains.py").read_text()
    assert "import" not in src.replace("from __future__ import annotations", "")


def test_design_and_optimizer_code_do_not_touch_brains():
    for name in ("schema.py", "bodylib.py", "body_template.py", "worlds.py", "catalog.py"):
        assert "brains" not in (ROOT / "dumbrobot" / name).read_text(), name

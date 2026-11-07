import numpy as np
import pytest

from dumbrobot import schema


def test_default_is_valid(genome):
    assert schema.validate(genome) == []
    assert len(genome) == schema.DIM


def same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == pytest.approx(b[k]) if isinstance(a[k], float) else a[k] == b[k], k


def test_unit_roundtrip_default(genome):
    same(schema.from_unit(schema.to_unit(genome)), genome)


def test_unit_roundtrip_random():
    rng = np.random.default_rng(1)
    for _ in range(50):
        g = schema.sample(rng)
        assert schema.validate(g) == []
        same(schema.from_unit(schema.to_unit(g)), g)


def test_from_unit_clips_and_covers_extremes():
    lo = schema.from_unit(np.full(schema.DIM, -5.0))
    hi = schema.from_unit(np.full(schema.DIM, 5.0))
    assert schema.validate(lo) == [] and schema.validate(hi) == []
    for p in schema.PARAMS:
        if p.kind != "enum":
            assert lo[p.name] == p.lo and hi[p.name] == p.hi
        else:
            assert lo[p.name] == p.choices[0] and hi[p.name] == p.choices[-1]


def test_unit_shape_checked():
    with pytest.raises(ValueError):
        schema.from_unit(np.zeros(3))


@pytest.mark.parametrize("key,val", [("wheel_r", 0.5), ("wheel_r", float("nan")), ("motor", "warp_drive"),
                                     ("whisker_n", 9), ("whisker_n", 1.5), ("bumper_on", True), ("wheel_r", "big")])
def test_bad_values_rejected_not_clipped(genome, key, val):
    genome[key] = val
    errs = schema.validate(genome)
    assert errs and key in errs[0]


def test_missing_and_unknown_keys(genome):
    del genome["wheel_r"]
    genome["laser"] = 1
    errs = " | ".join(schema.validate(genome))
    assert "missing parameter 'wheel_r'" in errs and "unknown parameter 'laser'" in errs


def test_describe_mentions_every_param():
    text = schema.describe()
    assert all(n in text for n in schema.NAMES)

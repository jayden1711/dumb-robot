"""Design-space schema shared by the LLM, random search and CMA-ES."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .catalog import CHASSIS_MATERIALS, MOTORS


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # "float" | "int" | "enum"
    default: Any
    lo: float | int | None = None
    hi: float | int | None = None
    choices: tuple = ()
    unit: str = ""


PARAMS: tuple[Param, ...] = (
    Param("chassis_len", "float", 0.20, 0.10, 0.35, unit="m"),
    Param("chassis_wid", "float", 0.14, 0.08, 0.25, unit="m"),
    Param("chassis_hgt", "float", 0.05, 0.02, 0.12, unit="m"),
    Param("chassis_mat", "enum", "abs_shell", choices=tuple(CHASSIS_MATERIALS)),
    Param("chassis_dz", "float", 0.02, -0.02, 0.06, unit="m"),  # chassis centre above axle height
    Param("wheel_r", "float", 0.04, 0.02, 0.09, unit="m"),
    Param("wheel_w", "float", 0.02, 0.01, 0.05, unit="m"),
    Param("axle_x", "float", 0.0, -0.08, 0.08, unit="m"),
    Param("wheel_mu", "float", 1.0, 0.3, 1.5),
    Param("motor", "enum", "tt_gear", choices=tuple(MOTORS)),
    Param("caster_f", "int", 1, 0, 1),
    Param("caster_r", "int", 1, 0, 1),
    Param("caster_rad", "float", 0.012, 0.008, 0.03, unit="m"),
    Param("caster_lift", "float", 0.004, 0.0, 0.03, unit="m"),  # caster bottom above ground at rest
    Param("bumper_on", "int", 0, 0, 1),
    Param("bumper_kind", "enum", "rigid", choices=("rigid", "switch")),
    Param("bumper_wid", "float", 0.8, 0.4, 1.2),  # fraction of chassis width
    Param("bumper_reach", "float", 0.03, 0.01, 0.08, unit="m"),
    Param("bumper_angle", "float", 0.0, -60.0, 60.0, unit="deg"),  # wedge half-angle, negative = funnel
    Param("bumper_hgt", "float", 0.03, 0.01, 0.06, unit="m"),
    Param("bumper_clear", "float", 0.01, 0.0, 0.08, unit="m"),  # bottom edge above ground
    Param("whisker_n", "int", 0, 0, 4),
    Param("whisker_len", "float", 0.12, 0.05, 0.25, unit="m"),
    Param("whisker_angle", "float", 40.0, 10.0, 80.0, unit="deg"),
    Param("whisker_k", "float", 0.02, 0.001, 0.1, unit="N m/rad"),
    Param("tail_len", "float", 0.0, 0.0, 0.25, unit="m"),
    Param("tail_drop", "float", 20.0, 0.0, 60.0, unit="deg"),
    Param("tail_mass", "float", 0.02, 0.01, 0.30, unit="kg"),
    Param("ballast_mass", "float", 0.0, 0.0, 0.5, unit="kg"),
    Param("ballast_x", "float", 0.0, -0.5, 0.5),  # fraction of half chassis length
    Param("ballast_z", "float", 0.0, -0.03, 0.05, unit="m"),
)
BY_NAME = {p.name: p for p in PARAMS}
NAMES = tuple(p.name for p in PARAMS)
DIM = len(PARAMS)


def default_genome() -> dict:
    return {p.name: p.default for p in PARAMS}


def validate(g: dict) -> list[str]:
    errs = []
    for k in g:
        if k not in BY_NAME:
            errs.append(f"unknown parameter '{k}'")
    for p in PARAMS:
        if p.name not in g:
            errs.append(f"missing parameter '{p.name}'")
            continue
        v = g[p.name]
        if p.kind == "enum":
            if v not in p.choices:
                errs.append(f"{p.name}={v!r} not in {list(p.choices)}")
        elif p.kind == "int":
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                errs.append(f"{p.name}={v!r} must be an int")
            elif not p.lo <= v <= p.hi:
                errs.append(f"{p.name}={v} outside [{p.lo}, {p.hi}]")
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)):
                errs.append(f"{p.name}={v!r} must be a number")
            elif not np.isfinite(v) or not p.lo <= v <= p.hi:
                errs.append(f"{p.name}={v} outside [{p.lo}, {p.hi}]")
    return errs


def from_unit(u) -> dict:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if u.shape != (DIM,):
        raise ValueError(f"expected shape ({DIM},), got {u.shape}")
    g = {}
    for x, p in zip(u, PARAMS):
        if p.kind == "float":
            g[p.name] = float(p.lo + x * (p.hi - p.lo))
        elif p.kind == "int":
            n = p.hi - p.lo + 1
            g[p.name] = int(p.lo + min(int(x * n), n - 1))
        else:
            n = len(p.choices)
            g[p.name] = p.choices[min(int(x * n), n - 1)]
    return g


def to_unit(g: dict) -> np.ndarray:
    errs = validate(g)
    if errs:
        raise ValueError("; ".join(errs))
    u = np.zeros(DIM)
    for i, p in enumerate(PARAMS):
        v = g[p.name]
        if p.kind == "float":
            u[i] = (v - p.lo) / (p.hi - p.lo)
        elif p.kind == "int":
            n = p.hi - p.lo + 1
            u[i] = (v - p.lo + 0.5) / n
        else:
            n = len(p.choices)
            u[i] = (p.choices.index(v) + 0.5) / n
    return u


def sample(rng: np.random.Generator) -> dict:
    return from_unit(rng.random(DIM))


def describe() -> str:
    """Schema table for designer prompts."""
    rows = []
    for p in PARAMS:
        rng = f"{list(p.choices)}" if p.kind == "enum" else f"[{p.lo}, {p.hi}]"
        rows.append(f"{p.name} ({p.kind}{', ' + p.unit if p.unit else ''}) default={p.default!r} range={rng}")
    return "\n".join(rows)

"""Load a body script, build robot + course into one spec, compile."""
from __future__ import annotations

import importlib.util
import itertools
from dataclasses import dataclass

import mujoco

from . import schema
from .bodylib import Builder, link_contacts

_counter = itertools.count()


class BuildError(Exception):
    pass


def load_script(path):
    """Return (genome, build_fn) from a body script file. Raises BuildError."""
    try:
        spec = importlib.util.spec_from_file_location(f"_body_{next(_counter)}", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return dict(mod.GENOME), mod.build
    except Exception as e:  # noqa: BLE001 - the script is untrusted input
        raise BuildError(f"{type(e).__name__}: {e}") from e


@dataclass
class Assembled:
    model: mujoco.MjModel
    builder: Builder
    course: object
    robot_geoms: list
    terrain_geoms: list


def _add_self_pairs(b):
    """Explicit pairs between robot geoms on different bodies (bypasses parent/child filtering), so the engine reports
    penetration depth between them. Used only by the validity gate's rest-pose check."""
    geoms = [b.spec.geom(n) for n, _ in b.collision_geoms]
    for i, g1 in enumerate(geoms):
        for g2 in geoms[i + 1:]:
            if g1.parent.name == g2.parent.name:
                continue
            if "chassis" in (g1.name, g2.name) and (g1.name.startswith("caster") or g2.name.startswith("caster")):
                continue  # the ball sits in a housing in the chassis (not modelled)
            p = b.spec.add_pair(geomname1=g1.name, geomname2=g2.name)
            p.condim = 3


def assemble(build_fn, genome, course, yaw=0.0, spawn_dz=0.01, self_pairs=False) -> Assembled:
    errs = schema.validate(genome)
    if errs:
        raise BuildError("schema: " + "; ".join(errs))
    b = Builder()
    try:
        build_fn(b, genome)
        if b.root is None or not b.actuators:
            raise BuildError("script must create a chassis and drive motors")
        terrain = course.add_to(b.spec)
        link_contacts(b.spec, b.collision_geoms, terrain)
        if self_pairs:
            _add_self_pairs(b)
        b.place(course.start_x, 0.0, yaw, spawn_dz)
        model = b.spec.compile()
    except BuildError:
        raise
    except Exception as e:  # noqa: BLE001
        raise BuildError(f"{type(e).__name__}: {e}") from e
    return Assembled(model, b, course, list(b.collision_geoms), terrain)

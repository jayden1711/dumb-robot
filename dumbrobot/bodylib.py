"""Part helpers on top of mujoco.MjSpec. A body script calls these; every part gets a real mass.

Frames: chassis body frame is x forward, y left, z up, origin at the chassis centre. Angles are radians in the spec.
Robot collision geoms use contype/conaffinity 1; terrain uses 2, so robot-terrain contacts exist only through the
explicit pairs added by `link_contacts` (geometric-mean friction).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import catalog as C

GEOM = mujoco.mjtGeom
ROBOT_BIT = 1
TERRAIN_BIT = 2
MIN_BODY_MASS = 1e-4


class Builder:
    def __init__(self):
        self.spec = mujoco.MjSpec()
        s = self.spec
        s.compiler.degree = False
        s.option.timestep = C.TIMESTEP
        s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.root = None
        self.zc = 0.0  # chassis centre height above ground at rest
        self.wheel_r = 0.0
        self.wheel_w = 0.0
        self.chassis_len = self.chassis_wid = self.chassis_hgt = 0.0
        self.motor = None
        self.collision_geoms: list[tuple[str, float]] = []  # (geom name, robot friction)
        self.actuators: list[str] = []
        self.wheel_joints: list[str] = []
        self.bumper_joint: str | None = None
        self.bumper_threshold = 0.003

    # ---- helpers -------------------------------------------------------
    def _geom(self, body, name, gtype, size, pos=(0, 0, 0), quat=None, euler=None, density=None, mass=None,
              mu=C.MU_BODY, collide=True, rgba=(0.6, 0.6, 0.6, 1)):
        kw = dict(name=name, type=gtype, size=list(size) + [0] * (3 - len(size)), pos=list(pos), rgba=list(rgba))
        if quat is not None:
            kw["quat"] = list(quat)
        if euler is not None:
            kw["euler"] = list(euler)
        if mass is not None:
            kw["mass"] = mass
        elif density is not None:
            kw["density"] = density
        if collide:
            kw["contype"] = ROBOT_BIT
            kw["conaffinity"] = ROBOT_BIT
            self.collision_geoms.append((name, mu))
        else:
            kw["contype"] = 0
            kw["conaffinity"] = 0
        return body.add_geom(**kw)

    def _mass_point(self, body, name, pos, mass, radius=0.008):
        return self._geom(body, name, GEOM.mjGEOM_SPHERE, [radius], pos=pos, mass=mass, collide=False,
                          rgba=(0.2, 0.2, 0.8, 1))

    # ---- parts ---------------------------------------------------------
    def add_chassis(self, length, width, height, density, wheel_r, dz):
        self.chassis_len, self.chassis_wid, self.chassis_hgt = length, width, height
        self.wheel_r = wheel_r
        self.zc = wheel_r + dz
        self.root = self.spec.worldbody.add_body(name="chassis", pos=[0, 0, self.zc])
        self.root.add_freejoint()
        self._geom(self.root, "chassis", GEOM.mjGEOM_BOX, [length / 2, width / 2, height / 2], density=density,
                   rgba=(0.8, 0.5, 0.1, 1))
        self._mass_point(self.root, "battery", (0, 0, 0), C.BATTERY_MASS, 0.015)
        return self.root

    def add_drive(self, wheel_r, wheel_w, axle_x, mu, motor_name):
        m = C.MOTORS[motor_name]
        self.motor, self.wheel_w = m, wheel_w
        y = self.chassis_wid / 2 + wheel_w / 2 + 0.002
        for side, sy in (("l", +1), ("r", -1)):
            body = self.root.add_body(name=f"wheel_{side}", pos=[axle_x, sy * y, self.wheel_r - self.zc])
            jn = f"wheel_{side}_joint"
            body.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=1e-3,
                           armature=m["armature"])
            self._geom(body, f"wheel_{side}", GEOM.mjGEOM_CYLINDER, [wheel_r, wheel_w / 2], euler=(math.pi / 2, 0, 0),
                       density=C.DENSITY_WHEEL, mu=mu, rgba=(0.1, 0.1, 0.1, 1))
            self._mass_point(self.root, f"motor_{side}", (axle_x, sy * (self.chassis_wid / 2 - 0.015), self.wheel_r - self.zc),
                             m["mass"], 0.01)
            a = self.spec.add_actuator(name=f"motor_{side}", target=jn)
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gainprm[0] = m["stall"]
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[2] = -m["stall"] / m["w0"]  # DC torque-speed line
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.ctrlrange = [-1.0, 1.0]
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [-m["stall"], m["stall"]]
            self.actuators.append(a.name)
            self.wheel_joints.append(jn)

    def add_caster(self, name, x, radius, lift=0.0):
        """Free-rolling ball transfer: a sphere on a ball joint; its bottom sits `lift` above the ground at rest."""
        z = radius + lift - self.zc
        body = self.root.add_body(name=name, pos=[x, 0, z])
        body.add_joint(name=f"{name}_joint", type=mujoco.mjtJoint.mjJNT_BALL, damping=2e-5, armature=2e-5)
        self._geom(body, name, GEOM.mjGEOM_SPHERE, [radius], density=2500.0, mu=C.MU_CASTER, rgba=(0.7, 0.7, 0.7, 1))

    def add_bumper(self, kind, width_frac, reach, angle, height, clear):
        """Two plates meeting at an apex ahead of the front face. angle>0: V pointing forward; <0: funnel."""
        L = width_frac * self.chassis_wid
        thick = 0.004
        xf = self.chassis_len / 2
        travel = 0.01
        z = -self.zc + clear + height / 2
        xa = xf + reach + (travel + 0.002 if kind == "switch" else 0.0)
        a = math.radians(angle)
        half_len = (L / 2) / math.cos(a)
        if kind == "switch":
            body = self.root.add_body(name="bumper", pos=[0, 0, 0])
            self.bumper_joint = "bumper_slide"
            body.add_joint(name=self.bumper_joint, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                           range=[-travel, 0.0], limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=150.0, damping=3.0,
                           armature=1e-4)
            self.spec.add_sensor(name="bumper_pos", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                                 objname=self.bumper_joint)
        else:
            body = self.root
        for side, sy in (("l", +1), ("r", -1)):
            # plate centre: half way from apex to the outer end
            cx = xa - (L / 4) * math.tan(a)
            cy = sy * L / 4
            yaw = sy * a  # outer ends lean back for a>0
            quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            self._geom(body, f"bump_{side}", GEOM.mjGEOM_BOX, [thick / 2, half_len / 2, height / 2], pos=(cx, cy, z), quat=quat,
                       density=C.DENSITY_BUMPER, rgba=(0.9, 0.2, 0.2, 1))

    def add_whiskers(self, n, length, angle, k):
        rad = 0.003
        xf = self.chassis_len / 2
        half_w = self.chassis_wid / 2
        slots = [(+1, xf + 0.002, half_w - 0.005, angle), (-1, xf + 0.002, half_w - 0.005, angle),
                 (+1, xf + 0.002, half_w / 2, 0.4 * angle), (-1, xf + 0.002, half_w / 2, 0.4 * angle)]
        for i in range(n):
            sy, x, y, ang = slots[i]
            yaw = sy * math.radians(ang)
            body = self.root.add_body(name=f"whisker_{i}", pos=[x, sy * y, 0.0],
                                      quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            body.add_joint(name=f"whisker_{i}_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-0.8, 0.8],
                           limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=k, damping=max(0.002, 0.1 * k), armature=2e-5)
            self._geom(body, f"whisker_{i}", GEOM.mjGEOM_CAPSULE, [rad, 0, 0], density=C.DENSITY_WHISKER,
                       euler=None, rgba=(0.9, 0.9, 0.2, 1)).fromto = [rad, 0, 0, rad + length, 0, 0]

    def add_tail(self, length, drop_deg, tip_mass):
        d = math.radians(drop_deg)
        x0, z0 = -self.chassis_len / 2, -self.chassis_hgt / 4
        x1, z1 = x0 - length * math.cos(d), z0 - length * math.sin(d)
        self._geom(self.root, "tail_rod", GEOM.mjGEOM_CAPSULE, [0.003, 0, 0], density=C.DENSITY_TAIL_ROD,
                   rgba=(0.3, 0.3, 0.3, 1)).fromto = [x0, 0, z0, x1, 0, z1]
        self._geom(self.root, "tail_tip", GEOM.mjGEOM_SPHERE, [0.008], pos=(x1, 0, z1), mass=tip_mass,
                   rgba=(0.3, 0.3, 0.3, 1))

    def add_ballast(self, mass, x, z):
        self._mass_point(self.root, "ballast", (x, 0, z), mass, 0.01)

    # ---- placement -----------------------------------------------------
    def place(self, x, y, yaw, dz=0.0):
        self.root.pos = [x, y, self.zc + dz]
        self.root.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def link_contacts(spec, robot_geoms, terrain_geoms):
    """Explicit robot x terrain pairs with geometric-mean friction."""
    for rn, rmu in robot_geoms:
        for tn, tmu in terrain_geoms:
            mu = math.sqrt(rmu * tmu)
            p = spec.add_pair(geomname1=rn, geomname2=tn)
            p.condim = 3
            p.friction = [mu, mu, 0.002, 1e-4, 1e-4]

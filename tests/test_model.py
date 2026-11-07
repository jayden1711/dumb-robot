import json

import mujoco
import numpy as np
import pytest

from dumbrobot import catalog as C
from dumbrobot import schema
from dumbrobot.assembly import BuildError, assemble
from dumbrobot.bodylib import Builder
from dumbrobot.evaluate import Ledger, evaluate
from dumbrobot.sim import run_episode
from dumbrobot.brains import BrainA
from dumbrobot.validity import check_design
from dumbrobot.worlds import flat_course, get_course


def names(rep):
    return {f["name"] for f in rep.failures}


# ---- motor model ----------------------------------------------------------
@pytest.mark.parametrize("motor", list(C.MOTORS))
def test_motor_torque_speed_line(template_build, genome, motor):
    genome["motor"] = motor
    asm = assemble(template_build, genome, flat_course())
    m = asm.model
    spec = C.MOTORS[motor]
    assert m.actuator_gainprm[0, 0] == pytest.approx(spec["stall"])
    assert m.actuator_biasprm[0, 2] == pytest.approx(-spec["stall"] / spec["w0"])
    assert tuple(m.actuator_forcerange[0]) == pytest.approx((-spec["stall"], spec["stall"]))
    d = mujoco.MjData(m)
    d.ctrl[:] = 1.0
    mujoco.mj_forward(m, d)
    assert d.actuator_force[0] == pytest.approx(spec["stall"])  # stall torque at zero speed
    d.ctrl[:] = 5.0  # command is clamped to [-1, 1]
    mujoco.mj_forward(m, d)
    assert d.actuator_force[0] == pytest.approx(spec["stall"])


def test_free_wheel_speed_matches_no_load_speed(template_build, genome):
    asm = assemble(template_build, genome, flat_course())
    m = asm.model
    d = mujoco.MjData(m)
    d.qpos[2] = 5.0  # hold the robot in the air so the wheels spin unloaded
    for _ in range(int(4.0 / m.opt.timestep)):
        d.qpos[2] = 5.0
        d.qvel[:6] = 0
        d.ctrl[:] = 1.0
        mujoco.mj_step(m, d)
    spec = C.MOTORS[genome["motor"]]
    bearing = 1e-3  # joint damping set by bodylib
    expected = spec["stall"] / (spec["stall"] / spec["w0"] + bearing)
    w = d.qvel[m.joint("wheel_l_joint").dofadr[0]]
    assert w == pytest.approx(expected, rel=0.02)
    assert w < spec["w0"]


def test_flat_ground_speed_below_wheel_limit(template_build, genome):
    asm = assemble(template_build, genome, flat_course())
    res = run_episode(asm, BrainA())
    vmax = C.MOTORS[genome["motor"]]["w0"] * genome["wheel_r"]
    assert 0.0 < res.progress * flat_course().length / res.time_s < vmax


def test_real_masses(template_build, genome):
    m = assemble(template_build, genome, flat_course()).model
    total = float(m.body_mass.sum())
    chassis_box = genome["chassis_len"] * genome["chassis_wid"] * genome["chassis_hgt"] * C.CHASSIS_MATERIALS[genome["chassis_mat"]]
    wheel = np.pi * genome["wheel_r"] ** 2 * genome["wheel_w"] * C.DENSITY_WHEEL
    floor = chassis_box + 2 * wheel + C.BATTERY_MASS + 2 * C.MOTORS[genome["motor"]]["mass"]
    assert total >= floor and total < 2 * floor


# ---- validity -------------------------------------------------------------
def test_default_design_valid(template_build, genome):
    assert check_design(template_build, genome).ok


def test_every_part_enabled_is_valid(template_build, genome):
    genome.update(bumper_on=1, bumper_kind="switch", bumper_angle=30.0, whisker_n=4, tail_len=0.08, tail_drop=15.0,
                  ballast_mass=0.1, ballast_x=0.3, chassis_dz=0.03)
    rep = check_design(template_build, genome)
    assert rep.ok, rep.text()


def test_rigid_wedge_bumper_valid(template_build, genome):
    genome.update(bumper_on=1, bumper_kind="rigid", bumper_angle=-40.0, bumper_clear=0.02)
    assert check_design(template_build, genome).ok


def test_reject_schema(template_build, genome):
    genome["wheel_r"] = 1.0
    assert names(check_design(template_build, genome)) == {"schema"}


def test_reject_build_error(genome):
    def bad(world, g):
        raise RuntimeError("boom")

    rep = check_design(bad, genome)
    assert names(rep) == {"build"} and "boom" in rep.failures[0]["detail"]


def test_reject_no_motors(genome):
    def no_drive(world, g):
        world.add_chassis(0.2, 0.14, 0.05, 300.0, 0.04, 0.02)

    assert names(check_design(no_drive, genome)) == {"build"}


def test_reject_too_heavy(template_build, genome):
    genome.update(chassis_len=0.35, chassis_wid=0.25, chassis_hgt=0.12, chassis_mat="alu_frame", ballast_mass=0.5)
    rep = check_design(template_build, genome)
    assert "mass" in names(rep)


def test_reject_oversize_envelope(template_build, genome):
    genome.update(chassis_len=0.35, tail_len=0.25, tail_drop=0.0)
    assert "envelope" in names(check_design(template_build, genome))


def test_reject_below_ground(template_build, genome):
    genome.update(tail_len=0.25, tail_drop=60.0)
    assert "ground_clip" in names(check_design(template_build, genome))


def test_reject_belly_in_ground(template_build, genome):
    genome.update(wheel_r=0.02, chassis_dz=-0.02, chassis_hgt=0.12)
    assert "ground_clip" in names(check_design(template_build, genome))


def test_reject_self_intersection(template_build, genome):
    def clipped(world, g):
        template_build(world, g)
        b = world.root.add_body(name="stray", pos=[0, 0, 0])
        b.add_joint(name="stray_j", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
        world._geom(b, "stray", mujoco.mjtGeom.mjGEOM_BOX, [0.05, 0.05, 0.02], pos=(0.0, 0.0, 0.0), density=300.0)

    rep = check_design(clipped, genome)
    assert "intersect" in names(rep)
    assert "stray" in rep.failures[0]["detail"]


def test_reject_reversed_motor_axis(template_build, genome):
    def mirrored(world, g):
        template_build(world, g)
        for j in ("wheel_l_joint", "wheel_r_joint"):
            world.spec.joint(j).axis = [0, -1, 0]

    assert "drive" in names(check_design(mirrored, genome))


def test_reject_unstable_at_rest(template_build, genome):
    genome.update(caster_f=0, caster_r=0, ballast_mass=0.5, ballast_x=0.5, chassis_hgt=0.02, chassis_dz=0.0, wheel_r=0.09)
    rep = check_design(template_build, genome)
    assert names(rep) == {"unstable"}, rep.text()


def test_reject_exploding_model(template_build, genome):
    def stiff(world, g):
        template_build(world, g)
        b = world.root.add_body(name="spring", pos=[0, 0, 0.05])
        b.add_joint(name="spring_j", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], stiffness=1e7, damping=0.0)
        world._geom(b, "spring", mujoco.mjtGeom.mjGEOM_SPHERE, [0.005], pos=(0.1, 0.0, 0.0), mass=0.001, collide=False)

    assert names(check_design(stiff, genome)) == {"explode"}


def test_random_genomes_never_crash_the_gate(template_build):
    rng = np.random.default_rng(7)
    for _ in range(15):
        rep = check_design(template_build, schema.sample(rng))
        assert rep.ok or rep.failures
        assert all(f["name"] in {"schema", "build", "mass", "envelope", "intersect", "ground_clip", "explode", "unstable",
                                 "drive"} for f in rep.failures)


# ---- courses / episodes ----------------------------------------------------
def test_assemble_rejects_bad_genome(template_build, genome):
    genome["motor"] = "nope"
    with pytest.raises(BuildError):
        assemble(template_build, genome, flat_course())


def test_friction_is_geometric_mean(template_build, genome):
    genome["wheel_mu"] = 1.2
    asm = assemble(template_build, genome, get_course("bumps"))
    m = asm.model
    for p in range(m.npair):
        if {m.geom(m.pair_geom1[p]).name, m.geom(m.pair_geom2[p]).name} == {"wheel_l", "floor"}:
            assert m.pair_friction[p][0] == pytest.approx(np.sqrt(1.2 * get_course("bumps").terrain_mu))
            return
    pytest.fail("wheel_l x floor pair not found")


def test_robot_and_terrain_only_collide_through_pairs(template_build, genome):
    m = assemble(template_build, genome, get_course("bumps")).model
    assert m.npair == len(assemble(template_build, genome, get_course("bumps")).robot_geoms) * 8


def test_episode_deterministic(big_design):
    g, b = big_design
    r1 = evaluate(b, g, seeds=(1,))
    r2 = evaluate(b, g, seeds=(1,))
    assert r1["score"] == r2["score"]
    assert r1["per_course"]["bumps"]["episodes"][0]["summary"] == r2["per_course"]["bumps"]["episodes"][0]["summary"]


def test_big_wheels_complete_course_one(big_design):
    g, b = big_design
    out = evaluate(b, g, courses=("bumps",), seeds=(1,))
    ep = out["per_course"]["bumps"]["episodes"][0]
    assert out["valid"] and ep["score"]["done"] and ep["result"]["status"] == "finished"
    assert ep["score"]["flips"] == 0
    assert all(s["status"] == "crossed" for s in ep["result"]["segments"])
    c = get_course("bumps")
    expected = 1.0 + 0.30 * (1 - ep["score"]["time_s"] / c.t_limit) + 0.15 * (1 - min(ep["score"]["energy_j"] / c.e_ref, 1))
    assert ep["score"]["score"] == pytest.approx(expected)
    assert ep["score"]["energy_j"] > 0


def test_small_wheels_fail_at_a_bump_with_a_usable_log(small_design):
    g, b = small_design
    out = evaluate(b, g, courses=("bumps",), seeds=(0,))
    ep = out["per_course"]["bumps"]["episodes"][0]
    res = ep["result"]
    assert out["valid"] and not ep["score"]["done"] and res["status"] in ("stuck", "timeout")
    failed = [s for s in res["segments"] if s["status"] == "failed_here"]
    assert len(failed) == 1 and failed[0]["name"].startswith("bump_")
    assert failed[0]["stuck_s"] > 0
    assert any(s["status"] == "not_reached" for s in res["segments"])
    assert 0.0 < ep["score"]["progress"] < 1.0
    assert ep["score"]["score"] == pytest.approx(ep["score"]["progress"])
    assert failed[0]["name"] in ep["summary"] and "touching" in ep["summary"]
    assert res["ending"]["contacts"]


def test_trace_recorded(big_design):
    g, b = big_design
    out = evaluate(b, g, seeds=(1,), record_trace=True)
    tr = out["per_course"]["bumps"]["episodes"][0]["trace"]
    assert tr["pos"].shape[1] == 3 and tr["quat"].shape[1] == 4 and tr["wheel"].shape[1] == 2
    assert np.all(np.diff(tr["t"]) > 0)


def test_invalid_design_not_simulated(template_build, genome):
    genome["tail_len"], genome["tail_drop"] = 0.25, 60.0
    out = evaluate(template_build, genome)
    assert not out["valid"] and out["score"] == 0.0 and out["sim_s"] == 0.0 and out["per_course"] == {}


# ---- budget ledger ---------------------------------------------------------
def test_ledger_charges_invalid_and_stops_at_budget(tmp_path, template_build, genome):
    led = Ledger(tmp_path / "ledger.jsonl", budget=2)
    bad = dict(genome, tail_len=0.25, tail_drop=60.0)
    assert led.charge("random", 0, evaluate(template_build, bad))
    assert led.charge("random", 0, evaluate(template_build, genome, courses=("flat",)))
    assert not led.charge("random", 0, evaluate(template_build, genome, courses=("flat",)))
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["valid"] is False and "ground_clip" in rows[0]["why_invalid"]
    assert rows[1]["valid"] is True and rows[1]["proposal"] == 2

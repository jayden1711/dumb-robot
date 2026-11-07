"""Default body script: maps the full genome onto bodylib parts. Designers copy and edit this file."""
from dumbrobot import catalog as C
from dumbrobot import schema

GENOME = schema.default_genome()


def build(world, g):
    world.add_chassis(g["chassis_len"], g["chassis_wid"], g["chassis_hgt"], C.CHASSIS_MATERIALS[g["chassis_mat"]],
                      g["wheel_r"], g["chassis_dz"])
    world.add_drive(g["wheel_r"], g["wheel_w"], g["axle_x"], g["wheel_mu"], g["motor"])
    inset = 0.015
    if g["caster_f"]:
        world.add_caster("caster_f", g["chassis_len"] / 2 - inset, g["caster_rad"], g["caster_lift"])
    if g["caster_r"]:
        world.add_caster("caster_r", -g["chassis_len"] / 2 + inset, g["caster_rad"], g["caster_lift"])
    if g["bumper_on"]:
        world.add_bumper(g["bumper_kind"], g["bumper_wid"], g["bumper_reach"], g["bumper_angle"], g["bumper_hgt"],
                         g["bumper_clear"] + 0.0)
    if g["whisker_n"]:
        world.add_whiskers(g["whisker_n"], g["whisker_len"], g["whisker_angle"], g["whisker_k"])
    if g["tail_len"] > 0:
        world.add_tail(g["tail_len"], g["tail_drop"], g["tail_mass"])
    if g["ballast_mass"] > 0:
        world.add_ballast(g["ballast_mass"], g["ballast_x"] * g["chassis_len"] / 2, g["ballast_z"])

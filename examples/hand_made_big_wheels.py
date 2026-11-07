"""Hand-made M0 design: large wheels, high-torque motor, two casters."""
from dumbrobot.body_template import build  # noqa: F401
from dumbrobot import schema

GENOME = schema.default_genome()
GENOME.update(wheel_r=0.09, wheel_w=0.03, motor="geared_550", chassis_dz=-0.01, caster_rad=0.015, caster_lift=0.006,
              chassis_len=0.22, chassis_wid=0.14)

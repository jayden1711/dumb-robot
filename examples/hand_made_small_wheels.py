"""Hand-made M0 contrast design: small wheels, so it should fail at an early bump."""
from dumbrobot.body_template import build  # noqa: F401
from dumbrobot import schema

GENOME = schema.default_genome()
GENOME.update(wheel_r=0.022, wheel_w=0.015, motor="tt_gear", caster_rad=0.01, chassis_dz=0.025, caster_lift=0.003)

"""Hand-written M1 design for the corridor: narrow chassis, forward-pointing rigid wedge, slippery wheels, tail and ballast."""
from dumbrobot.body_template import build  # noqa: F401
from dumbrobot import schema

GENOME = schema.default_genome()
GENOME.update(chassis_len=0.18, chassis_wid=0.085, chassis_hgt=0.054, chassis_mat="foam_shell", chassis_dz=0.05, wheel_r=0.07,
              wheel_w=0.01, axle_x=-0.08, wheel_mu=0.33, motor="geared_550", caster_f=0, caster_r=0, caster_lift=0.028,
              bumper_on=1, bumper_kind="rigid", bumper_wid=0.51, bumper_reach=0.06, bumper_angle=55.0, bumper_hgt=0.055,
              bumper_clear=0.012, tail_len=0.11, tail_drop=38.5, tail_mass=0.28, ballast_mass=0.38, ballast_x=0.34,
              ballast_z=-0.022)

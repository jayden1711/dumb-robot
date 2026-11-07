"""Motor and material catalogs. Approximate engineering values, not from any datasheet [verify]."""

# stall torque [N m], no-load speed [rad/s], mass [kg], reflected rotor armature [kg m^2]
MOTORS = {
    "micro_fast": dict(stall=0.04, w0=30.0, mass=0.012, armature=2e-4),
    "micro_torque": dict(stall=0.12, w0=10.0, mass=0.015, armature=4e-4),
    "tt_gear": dict(stall=0.15, w0=20.0, mass=0.050, armature=6e-4),
    "geared_550": dict(stall=1.00, w0=15.0, mass=0.250, armature=2e-3),
}

# effective density of the finished chassis volume [kg/m^3] (shelled / framed, so below bulk density)
CHASSIS_MATERIALS = {
    "foam_shell": 120.0,
    "abs_shell": 300.0,
    "alu_frame": 700.0,
}

DENSITY_WHEEL = 600.0  # rubber tyre on a hub, effective
DENSITY_WHISKER = 1200.0
DENSITY_BUMPER = 400.0  # plastic plate, effective
DENSITY_TAIL_ROD = 1500.0  # carbon/glass rod, effective
MU_CASTER = 0.8  # grip of the rolling ball on terrain (it rolls, so this is not drag)
MU_BODY = 0.5  # unpadded chassis, bumper, whisker, tail rod
BATTERY_MASS = 0.20  # fixed electronics + battery, mounted at the chassis centre

TIMESTEP = 0.002
CONTROL_DT = 0.01
G = 9.81

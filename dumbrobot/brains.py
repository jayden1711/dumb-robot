"""Frozen brains. Parameter-free; they never see the genome, the course or the clock beyond their own reflex timer."""
from __future__ import annotations


class BrainA:
    """Both motors full forward, no sensors."""

    name = "A"

    def reset(self):
        pass

    def act(self, t: float, sensors: dict) -> tuple[float, float]:
        return (1.0, 1.0)


class BrainB:
    """One bumper reflex: pressed -> reverse while turning for 0.5 s, then forward again. Fixed turn direction."""

    name = "B"
    REFLEX_S = 0.5

    def __init__(self):
        self.reset()

    def reset(self):
        self._until = -1.0

    def act(self, t: float, sensors: dict) -> tuple[float, float]:
        if sensors.get("bumper", False) and t >= self._until:
            self._until = t + self.REFLEX_S
        if t < self._until:
            return (-1.0, -0.3)
        return (1.0, 1.0)


BRAINS = {"A": BrainA, "B": BrainB}

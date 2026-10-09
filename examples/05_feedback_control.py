"""An active experiment: a consumer acts, and the action changes what is observed next.

    observation -> procedure -> output -> controller -> actuator -> plant -> observation

The filter never actuates anything.  The controller is a consumer; it acts
on the plant, not on the run.  The action it applied comes back through
the statistical boundary as a declared logical input `u`, paired with the
measurement it influenced, so the filter's model of the plant stays correct
and the action is part of the recorded history.
"""
from seqinfer import Observation, PositionalPair, Recorder, Run, SplitMix64, run_sync
from seqinfer.procedures import LocalLevelKalman

SETPOINT, GAIN, STEPS = 5.0, 0.4, 40


class Plant:
    """A drifting level that can be nudged by an actuator."""

    def __init__(self, seed: int):
        self.level, self.command = 0.0, 0.0
        self.rng = SplitMix64.for_role(seed, "simulation/plant")

    def actuate(self, u: float) -> None:
        self.command = u

    def observations(self, n: int):
        for i in range(n):
            u, self.command = self.command, 0.0
            self.level += u + self.rng.normal(0.0, 0.1)
            yield Observation("u", u, source="actuator", seq=i)
            yield Observation("y", self.level + self.rng.normal(0.0, 0.5), source="sensor", seq=i)


plant = Plant(seed=1)
recorder = Recorder()


def controller(event):
    plant.actuate(GAIN * (SETPOINT - event.output.mean))


run = Run(
    LocalLevelKalman(q=0.01, r=0.25, control=True),
    topology=PositionalPair(("u", "y")),
    consumers=[recorder, controller],
)
run_sync(run, plant.observations(STEPS))

for out in recorder.outputs[::5] + recorder.outputs[-1:]:
    print(f"t={out.n:2d}  estimate={out.mean:6.3f} +/- {out.var ** 0.5:.3f}")
print(f"true level {plant.level:.3f}, setpoint {SETPOINT}")

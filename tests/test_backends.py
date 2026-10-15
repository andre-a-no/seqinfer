# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""The numpy backend: same mathematics as the Python one, and a checkpoint never switches between them."""
import random
import unittest
from unittest import mock

from sequential_analysis import IncompatibleCheckpoint, Run, backend
from sequential_analysis.procedures import (
    BettingConfidenceSequence,
    CatoniConfidenceSequence,
    KernelTwoSampleTest,
    ParticleFilter,
    StateSpaceModel,
    StochasticVolatilityModel,
)
from sequential_analysis.rng import SplitMix64

HAVE_NUMPY = backend.np is not None


def outputs(proc, inputs, rng=None):
    state, outs = proc.initial_state(), []
    for x in inputs:
        state, out = proc.step(state, x, rng)
        outs.append(out)
    return state, outs


def close(a, b, rel):
    for field in a.__dataclass_fields__:
        x, y = getattr(a, field), getattr(b, field)
        if isinstance(x, float) and isinstance(y, float):
            if abs(x - y) > rel * max(1.0, abs(x)):
                return False
        elif x != y:
            return False
    return True


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class NumpyAgreesWithPython(unittest.TestCase):
    def setUp(self):
        rnd = random.Random(1)
        self.pairs = [{"a": rnd.gauss(0, 1), "b": rnd.gauss(0.3, 1.5)} for _ in range(300)]
        self.bounded = [{"x": 10 * rnd.random()} for _ in range(300)]

    def check(self, make, inputs, rel, rng_seed=None):
        rng = (lambda: SplitMix64.for_role(rng_seed, "test")) if rng_seed is not None else (lambda: None)
        py_rng, np_rng = rng(), rng()
        _, py = outputs(make("python"), inputs, py_rng)
        _, nu = outputs(make("numpy"), inputs, np_rng)
        for t, (a, b) in enumerate(zip(py, nu, strict=True)):
            self.assertTrue(close(a, b, rel), f"step {t}: {a} != {b}")
        if py_rng is not None:
            self.assertEqual(py_rng.state, np_rng.state)  # the same random numbers were drawn

    def test_kernel_two_sample(self):
        self.check(lambda b: KernelTwoSampleTest(alpha=1e-9, backend=b), self.pairs, 1e-12)

    def test_betting_confidence_sequence(self):
        self.check(lambda b: BettingConfidenceSequence(0.0, 10.0, grid=300, backend=b), self.bounded, 1e-12)

    def test_catoni_confidence_sequence(self):
        self.check(lambda b: CatoniConfidenceSequence(3.0, backend=b), self.bounded[:150], 1e-9)

    def test_particle_filter(self):
        ys = [{"y": v} for v in (0.1, -0.3, 0.5, 0.2, -0.1, 1.5, -0.7, 0.05)]
        self.check(lambda b: ParticleFilter(StochasticVolatilityModel(-1.0, 0.95, 0.3), 2000, backend=b), ys, 1e-9,
                   rng_seed=4)

    def test_vectorised_draws_are_the_scalar_draws(self):
        a = SplitMix64.for_role(7, "x")
        b = a.copy()
        self.assertEqual(backend.VectorRandom(a).random(1000).tolist(), [b.random() for _ in range(1000)])
        self.assertEqual(a.state, b.state)


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class BackendIsPartOfTheIdentity(unittest.TestCase):
    def test_auto_resolves_and_is_recorded(self):
        self.assertEqual(KernelTwoSampleTest().config()["backend"], "numpy")
        self.assertEqual(KernelTwoSampleTest(backend="python").config()["backend"], "python")

    def test_a_checkpoint_does_not_change_backend(self):
        run = Run(BettingConfidenceSequence(grid=50, backend="numpy"))
        run.step({"x": 0.4})
        checkpoint = run.checkpoint()
        with self.assertRaises(IncompatibleCheckpoint):
            Run.restore(BettingConfidenceSequence(grid=50, backend="python"), checkpoint)
        Run.restore(BettingConfidenceSequence(grid=50, backend="numpy"), checkpoint)
        self.assertIsNotNone(run.provenance["numerics"]["numpy"])

    def test_a_model_without_vectorised_methods_runs_in_python(self):
        class Walk(StateSpaceModel):
            name = "walk"

            def config(self):
                return {}

            def initial(self, rng):
                return 0.0

            def transition(self, state, rng):
                return rng.normal(state, 1.0)

            def log_likelihood(self, state, y):
                return -0.5 * (y - state) ** 2

        self.assertEqual(ParticleFilter(Walk(), 10).backend, "python")
        with self.assertRaises(ValueError):
            ParticleFilter(Walk(), 10, backend="numpy")


class WithoutNumpy(unittest.TestCase):
    def test_auto_falls_back_and_numpy_is_refused(self):
        with mock.patch.object(backend, "np", None):
            self.assertEqual(backend.resolve("auto"), "python")
            self.assertEqual(KernelTwoSampleTest().backend, "python")
            with self.assertRaises(ValueError):
                KernelTwoSampleTest(backend="numpy")
        with self.assertRaises(ValueError):
            backend.resolve("fortran")


if __name__ == "__main__":
    unittest.main()

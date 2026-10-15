# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Novikov Laboratories LLC (Kazan, Tatarstan, Russian Federation)
# Commercial licenses for use outside the AGPL: see COMMERCIAL.md
"""Reference procedures.  Each one is only a transition function and its state."""
from .ab import ABTest, BoundedABTest, MultipleTests, SampleRatioMismatch, TwoProportionTest, e_bh
from .betting import BettingConfidenceSequence, BettingMeanTest, SequentialSignTest
from .bocpd import BayesianChangePoint, BetaBernoulliModel, NormalGammaModel, PoissonGammaModel
from .cusum import CUSUM
from .edetector import MeanEDetector, QuantileEDetector
from .ema import EMA
from .families import Bernoulli, Exponential, Family, Gaussian, Poisson
from .group_sequential import GroupSequentialTest
from .hmm import CategoricalEmission, GaussianEmission, HiddenMarkovFilter, PoissonEmission
from .kalman import LocalLevelKalman
from .mixture import NormalMixtureSPRT
from .online_fdr import OnlineFDR
from .particle import BootstrapParticleFilter
from .particle_filter import (
    LinearGaussianModel,
    ParticleFilter,
    StateSpaceModel,
    StochasticVolatilityModel,
    StudentTLocalLevel,
)
from .plan import PlanTest
from .quantile import QuantileConfidenceSequence, QuantileTest
from .rates import RateRatioTest
from .robust_mean import AsymptoticConfidenceSequence, CatoniConfidenceSequence
from .shiryaev_roberts import ShiryaevRoberts
from .sprt import ACCEPT_H0, REJECT_H0, SPRT
from .state_space import KalmanFilter, RobustKalmanFilter
from .t_mixture import TMixtureSPRT
from .two_sample import MeanDifference
from .two_sample_betting import KernelTwoSampleTest
from .two_sprt import TwoSPRT, kiefer_weiss_point

__all__ = [
    "ACCEPT_H0",
    "CUSUM",
    "EMA",
    "REJECT_H0",
    "SPRT",
    "ABTest",
    "AsymptoticConfidenceSequence",
    "BayesianChangePoint",
    "Bernoulli",
    "BetaBernoulliModel",
    "BettingConfidenceSequence",
    "BettingMeanTest",
    "BootstrapParticleFilter",
    "BoundedABTest",
    "CategoricalEmission",
    "CatoniConfidenceSequence",
    "Exponential",
    "Family",
    "Gaussian",
    "GaussianEmission",
    "GroupSequentialTest",
    "HiddenMarkovFilter",
    "KalmanFilter",
    "KernelTwoSampleTest",
    "LinearGaussianModel",
    "LocalLevelKalman",
    "MeanDifference",
    "MeanEDetector",
    "MultipleTests",
    "NormalGammaModel",
    "NormalMixtureSPRT",
    "OnlineFDR",
    "ParticleFilter",
    "PlanTest",
    "Poisson",
    "PoissonEmission",
    "PoissonGammaModel",
    "QuantileConfidenceSequence",
    "QuantileEDetector",
    "QuantileTest",
    "RateRatioTest",
    "RobustKalmanFilter",
    "SampleRatioMismatch",
    "SequentialSignTest",
    "ShiryaevRoberts",
    "StateSpaceModel",
    "StochasticVolatilityModel",
    "StudentTLocalLevel",
    "TMixtureSPRT",
    "TwoProportionTest",
    "TwoSPRT",
    "e_bh",
    "kiefer_weiss_point",
]

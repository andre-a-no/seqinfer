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
from .forecast import (
    AutoRegressive,
    ConformalForecast,
    ExponentialSmoothing,
    ForecastComparison,
    ModelAveraging,
    averaged_smoothing,
)
from .group_sequential import (
    GroupSequentialLogRank,
    GroupSequentialMonitor,
    GroupSequentialProportions,
    GroupSequentialTest,
    GroupSequentialTwoMeans,
)
from .hmm import (
    CategoricalEmission,
    GaussianEmission,
    HiddenMarkovFilter,
    MultivariateGaussianEmission,
    PoissonEmission,
    StudentTEmission,
    emission_from_spec,
)
from .kalman import LocalLevelKalman
from .linear_model import AnytimeLinearModel
from .mixture import NormalMixtureSPRT
from .online_fdr import OnlineFDR
from .online_hmm import OnlineHMM
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
from .ratio import RatioABTest
from .robust_mean import AsymptoticConfidenceSequence, CatoniConfidenceSequence
from .shiryaev_roberts import ShiryaevRoberts
from .sprt import ACCEPT_H0, REJECT_H0, SPRT
from .state_space import KalmanFilter, RobustKalmanFilter
from .survival import AnytimeLogRankTest
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
    "AnytimeLinearModel",
    "AnytimeLogRankTest",
    "AsymptoticConfidenceSequence",
    "AutoRegressive",
    "BayesianChangePoint",
    "Bernoulli",
    "BetaBernoulliModel",
    "BettingConfidenceSequence",
    "BettingMeanTest",
    "BootstrapParticleFilter",
    "BoundedABTest",
    "CategoricalEmission",
    "CatoniConfidenceSequence",
    "ConformalForecast",
    "Exponential",
    "ExponentialSmoothing",
    "Family",
    "ForecastComparison",
    "Gaussian",
    "GaussianEmission",
    "GroupSequentialLogRank",
    "GroupSequentialMonitor",
    "GroupSequentialProportions",
    "GroupSequentialTest",
    "GroupSequentialTwoMeans",
    "HiddenMarkovFilter",
    "KalmanFilter",
    "KernelTwoSampleTest",
    "LinearGaussianModel",
    "LocalLevelKalman",
    "MeanDifference",
    "MeanEDetector",
    "ModelAveraging",
    "MultipleTests",
    "MultivariateGaussianEmission",
    "NormalGammaModel",
    "NormalMixtureSPRT",
    "OnlineFDR",
    "OnlineHMM",
    "ParticleFilter",
    "PlanTest",
    "Poisson",
    "PoissonEmission",
    "PoissonGammaModel",
    "QuantileConfidenceSequence",
    "QuantileEDetector",
    "QuantileTest",
    "RateRatioTest",
    "RatioABTest",
    "RobustKalmanFilter",
    "SampleRatioMismatch",
    "SequentialSignTest",
    "ShiryaevRoberts",
    "StateSpaceModel",
    "StochasticVolatilityModel",
    "StudentTEmission",
    "StudentTLocalLevel",
    "TMixtureSPRT",
    "TwoProportionTest",
    "TwoSPRT",
    "averaged_smoothing",
    "e_bh",
    "emission_from_spec",
    "kiefer_weiss_point",
]

"""wellmodel - an open, scriptable oil-well modelling toolkit.

Core:          BlackOilPVT, Wellbore, WellModel (vectorised VLP traverse + nodal)
Workflow:      qc_tests -> compare_correlations -> match_tests
Uncertainty:   monte_carlo with Dist classes (+ match covariance)
Transient:     RadialReservoir, fit_buildup, forecast
"""
from .pvt import BlackOilPVT
from .wellbore import Wellbore
from .model import WellModel, OperatingPoint
from .qc import qc_tests, QCConfig, QCReport
from .matching import match_tests, residual_table, MatchResult, ACCEPTABLE
from .selection import compare_correlations, SelectionResult
from .probabilistic import monte_carlo, Fixed, Uniform, Triangular, Normal, LogNormal, MCResult
from .transient import (RadialReservoir, fit_buildup, bourdet_derivative, forecast,
                        DepletionInputs, logistic_water_cut, solution_gas_gor)
from .correlations import CORRELATIONS, LABELS

__version__ = "0.1.0"

"""
Comprehensive testing framework for LangCoop simulations.
Python 3.12 + CARLA 0.9.16+ with leaderboard-style evaluation.

Features:
- Route Completion (RC) and Driving Score (DS) metrics
- Scenario management with custom routes
- Multi-agent evaluation
- Matplotlib visualization and reporting
"""

from .test_runner import TestRunner
from .evaluator.metrics import MetricsCalculator, RouteCompletionMetrics
from .evaluator.scenario_manager import ScenarioManager, Route, Scenario
from .evaluator.scenario_evaluator import ScenarioEvaluator
from .agents import SimpleAutonomousAgent
from .controllers import PIDController

__version__ = "0.2.0"

__all__ = [
    "TestRunner",
    "ScenarioEvaluator",
    "MetricsCalculator",
    "RouteCompletionMetrics",
    "ScenarioManager",
    "Route",
    "Scenario",
    "SimpleAutonomousAgent",
    "PIDController",
]

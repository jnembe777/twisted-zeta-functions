"""
Experiments module for running computational studies.

Contains:
- config: Global parameters loading
- runner: Parallel execution orchestrator
- validator: Reproducibility tests
- population: Definition of 60 arithmetics
"""

from arithmetics.experiments.runner import run_experiment, run_all, ExperimentRunner
from arithmetics.experiments.validator import (
    validate_reproducibility,
    validate_precision_convergence,
    validate_classical_case,
)
from arithmetics.experiments.population import get_all_arithmetics, get_strata

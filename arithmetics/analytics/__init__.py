"""
Analytics module for derived quantities and visualization.

Contains:
- chebyshev: ψ_φ(x) twisted Chebyshev function
- distribution: Δ_φ(X) prime distribution defect
- visualization: Heatmaps, zero trajectories
"""

from arithmetics.analytics.chebyshev import psi_phi, von_mangoldt, explicit_formula
from arithmetics.analytics.distribution import (
    delta_distribution,
    compare_prime_sets,
    growth_rate_analysis,
)
from arithmetics.analytics.visualization import (
    plot_zero_distribution,
    plot_defect_heatmap,
    plot_zero_trajectory,
)

"""
Prime distribution defect analysis.

Computes Δ_φ(X), the prime distribution defect that measures how the
distribution of φ-twisted primes differs from classical primes.

Δ_φ(X) = Σ_{p≤X} |log(φ(p)/p)| + #{p ∈ P_φ △ P : p ≤ X}

where P_φ △ P is the symmetric difference of twisted and classical primes.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
from mpmath import mp, mpf, log as mp_log, fabs
import numpy as np
from scipy import stats

from arithmetics.core.transfer import ArithmeticTransfer, ExponentialTransfer
from arithmetics.zeta.euler_product import sieve_primes, twisted_primes


def delta_distribution(
    phi: ArithmeticTransfer,
    X: int,
    precision: int = 30
) -> Dict[str, Any]:
    """
    Compute the prime distribution defect Δ_φ(X).

    Δ_φ(X) = Σ_{p≤X} |log(φ(p)/p)| + #{p ∈ P_φ △ P : p ≤ X}

    Args:
        phi: The arithmetic transfer
        X: Upper limit for primes
        precision: Decimal places

    Returns:
        Dictionary with defect components and total
    """
    mp.dps = precision
    phi.precision = precision

    # Get classical and twisted primes
    classical = set(sieve_primes(X))
    twisted = set(twisted_primes(phi, X, precision))

    # Symmetric difference
    sym_diff = classical.symmetric_difference(twisted)
    sym_diff_count = len(sym_diff)

    # Log ratio sum over classical primes
    log_ratio_sum = mpf(0)
    log_ratios = []

    for p in classical:
        phi_p = phi.phi(p)
        if phi_p > 0:
            ratio = phi_p / mpf(p)
            log_ratio = fabs(mp_log(ratio))
            log_ratio_sum += log_ratio
            log_ratios.append((p, float(log_ratio)))

    # Total defect
    delta = float(log_ratio_sum) + sym_diff_count

    return {
        'X': X,
        'delta_phi_X': delta,
        'log_ratio_component': float(log_ratio_sum),
        'symmetric_difference_component': sym_diff_count,
        'n_classical_primes': len(classical),
        'n_twisted_primes': len(twisted),
        'n_in_symmetric_difference': sym_diff_count,
        'only_classical': sorted([p for p in sym_diff if p in classical])[:20],
        'only_twisted': sorted([p for p in sym_diff if p in twisted])[:20],
        'top_log_ratios': sorted(log_ratios, key=lambda x: -x[1])[:10],
    }


def compare_prime_sets(
    phi: ArithmeticTransfer,
    X: int,
    precision: int = 30
) -> Dict[str, Any]:
    """
    Detailed comparison of classical and twisted prime sets.

    Args:
        phi: The arithmetic transfer
        X: Upper limit
        precision: Decimal places

    Returns:
        Comparison statistics
    """
    mp.dps = precision
    phi.precision = precision

    classical = set(sieve_primes(X))
    twisted = set(twisted_primes(phi, X, precision))

    intersection = classical & twisted
    only_classical = classical - twisted
    only_twisted = twisted - classical
    union = classical | twisted

    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 1.0

    # Compute distribution statistics
    classical_list = sorted(classical)
    twisted_list = sorted(twisted)

    # Gaps between consecutive primes
    classical_gaps = [classical_list[i+1] - classical_list[i]
                      for i in range(len(classical_list)-1)]
    twisted_gaps = [twisted_list[i+1] - twisted_list[i]
                    for i in range(len(twisted_list)-1)] if len(twisted_list) > 1 else []

    return {
        'X': X,
        'n_classical': len(classical),
        'n_twisted': len(twisted),
        'n_intersection': len(intersection),
        'n_only_classical': len(only_classical),
        'n_only_twisted': len(only_twisted),
        'jaccard_similarity': jaccard,
        'density_classical': len(classical) / X,
        'density_twisted': len(twisted) / X,
        'mean_gap_classical': np.mean(classical_gaps) if classical_gaps else 0,
        'mean_gap_twisted': np.mean(twisted_gaps) if twisted_gaps else 0,
        'max_gap_classical': max(classical_gaps) if classical_gaps else 0,
        'max_gap_twisted': max(twisted_gaps) if twisted_gaps else 0,
        'smallest_only_classical': min(only_classical) if only_classical else None,
        'smallest_only_twisted': min(only_twisted) if only_twisted else None,
    }


def growth_rate_analysis(
    phi: ArithmeticTransfer,
    x_values: List[int] = None,
    precision: int = 30
) -> Dict[str, Any]:
    """
    Analyze the growth rate of Δ_φ(X) as X increases.

    Args:
        phi: The arithmetic transfer
        x_values: Values of X to test
        precision: Decimal places

    Returns:
        Growth rate analysis
    """
    if x_values is None:
        x_values = [10, 100, 1000, 10000, 100000]

    mp.dps = precision
    phi.precision = precision

    results = []
    for X in x_values:
        delta_result = delta_distribution(phi, X, precision)
        results.append({
            'X': X,
            'delta': delta_result['delta_phi_X'],
            'log_X': np.log(X),
            'log_delta': np.log(delta_result['delta_phi_X']) if delta_result['delta_phi_X'] > 0 else float('-inf'),
        })

    # Fit growth rate: Δ(X) ~ X^α or Δ(X) ~ (log X)^β
    valid_results = [r for r in results if r['log_delta'] > float('-inf')]

    if len(valid_results) >= 2:
        log_x = np.array([r['log_X'] for r in valid_results])
        log_delta = np.array([r['log_delta'] for r in valid_results])

        # Linear fit: log(Δ) = α·log(X) + c
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_delta)

        growth_exponent = slope
        r_squared = r_value ** 2
    else:
        growth_exponent = None
        r_squared = None

    return {
        'transfer_name': phi.name,
        'x_values': x_values,
        'delta_values': [r['delta'] for r in results],
        'growth_exponent': growth_exponent,
        'r_squared': r_squared,
        'interpretation': _interpret_growth(growth_exponent) if growth_exponent else "Insufficient data",
        'detailed_results': results,
    }


def _interpret_growth(exponent: float) -> str:
    """Interpret the growth exponent."""
    if exponent < 0.1:
        return f"Δ(X) grows very slowly (≈ constant or logarithmic)"
    elif exponent < 0.5:
        return f"Δ(X) grows sublinearly: O(X^{exponent:.3f})"
    elif exponent < 1.0:
        return f"Δ(X) grows sublinearly: O(X^{exponent:.3f})"
    elif exponent < 1.1:
        return f"Δ(X) grows approximately linearly: O(X)"
    else:
        return f"Δ(X) grows superlinearly: O(X^{exponent:.3f})"


def defect_by_prime_size(
    phi: ArithmeticTransfer,
    X: int,
    n_bins: int = 10,
    precision: int = 30
) -> Dict[str, Any]:
    """
    Analyze how the defect varies with prime size.

    Args:
        phi: The arithmetic transfer
        X: Upper limit
        n_bins: Number of bins
        precision: Decimal places

    Returns:
        Binned defect analysis
    """
    mp.dps = precision
    phi.precision = precision

    primes = sieve_primes(X)

    # Compute log(φ(p)/p) for each prime
    defects = []
    for p in primes:
        phi_p = phi.phi(p)
        if phi_p > 0:
            log_ratio = float(fabs(mp_log(phi_p / mpf(p))))
            defects.append((p, log_ratio))

    if not defects:
        return {'error': 'No valid defects computed'}

    # Bin by prime size
    prime_vals = np.array([d[0] for d in defects])
    defect_vals = np.array([d[1] for d in defects])

    bin_edges = np.logspace(np.log10(min(prime_vals)), np.log10(max(prime_vals)), n_bins + 1)
    bin_indices = np.digitize(prime_vals, bin_edges)

    binned_results = []
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if np.any(mask):
            binned_results.append({
                'bin_min': float(bin_edges[i-1]),
                'bin_max': float(bin_edges[i]),
                'n_primes': int(np.sum(mask)),
                'mean_defect': float(np.mean(defect_vals[mask])),
                'max_defect': float(np.max(defect_vals[mask])),
                'min_defect': float(np.min(defect_vals[mask])),
            })

    return {
        'X': X,
        'n_bins': n_bins,
        'n_primes_total': len(primes),
        'overall_mean_defect': float(np.mean(defect_vals)),
        'binned_results': binned_results,
    }


def correlation_with_cohomology(
    transfers: List[ArithmeticTransfer],
    X: int = 10000,
    precision: int = 30
) -> Dict[str, Any]:
    """
    Analyze correlation between distribution defect and cohomological norm.

    Args:
        transfers: List of arithmetic transfers to compare
        X: Upper limit for prime comparison
        precision: Decimal places

    Returns:
        Correlation analysis
    """
    from arithmetics.core.cohomology import compute_regular_norm

    mp.dps = precision

    data_points = []

    for phi in transfers:
        phi.precision = precision

        # Compute distribution defect
        delta_result = delta_distribution(phi, X, precision)
        delta = delta_result['delta_phi_X']

        # Compute cohomological norm
        regular_norm = float(compute_regular_norm(phi, precision=precision))

        data_points.append({
            'name': phi.name,
            'delta': delta,
            'regular_norm': regular_norm,
        })

    # Compute correlation
    if len(data_points) >= 2:
        deltas = np.array([d['delta'] for d in data_points])
        norms = np.array([d['regular_norm'] for d in data_points])

        # Filter out any inf/nan
        valid = np.isfinite(deltas) & np.isfinite(norms)
        if np.sum(valid) >= 2:
            correlation, p_value = stats.pearsonr(deltas[valid], norms[valid])
            spearman_corr, spearman_p = stats.spearmanr(deltas[valid], norms[valid])
        else:
            correlation, p_value = None, None
            spearman_corr, spearman_p = None, None
    else:
        correlation, p_value = None, None
        spearman_corr, spearman_p = None, None

    return {
        'X': X,
        'n_transfers': len(transfers),
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'data_points': data_points,
    }


class DistributionAnalyzer:
    """
    Comprehensive analyzer for prime distribution under arithmetic transfers.
    """

    def __init__(
        self,
        phi: ArithmeticTransfer,
        precision: int = 30
    ):
        """
        Initialize the analyzer.

        Args:
            phi: The arithmetic transfer
            precision: Decimal places
        """
        self.phi = phi
        self.precision = precision
        self._classical_primes_cache = {}
        self._twisted_primes_cache = {}

        mp.dps = precision
        phi.precision = precision

    def get_classical_primes(self, X: int) -> Set[int]:
        """Get classical primes up to X with caching."""
        if X not in self._classical_primes_cache:
            self._classical_primes_cache[X] = set(sieve_primes(X))
        return self._classical_primes_cache[X]

    def get_twisted_primes(self, X: int) -> Set[int]:
        """Get twisted primes up to X with caching."""
        if X not in self._twisted_primes_cache:
            self._twisted_primes_cache[X] = set(twisted_primes(self.phi, X, self.precision))
        return self._twisted_primes_cache[X]

    def full_analysis(self, X: int) -> Dict[str, Any]:
        """
        Perform complete distribution analysis at scale X.

        Args:
            X: Upper limit

        Returns:
            Complete analysis dictionary
        """
        delta = delta_distribution(self.phi, X, self.precision)
        comparison = compare_prime_sets(self.phi, X, self.precision)
        binned = defect_by_prime_size(self.phi, X, precision=self.precision)

        return {
            'transfer_name': self.phi.name,
            'X': X,
            'delta_analysis': delta,
            'set_comparison': comparison,
            'binned_analysis': binned,
        }

    def growth_analysis(self, x_max: int = 100000) -> Dict[str, Any]:
        """
        Analyze growth of distribution defect.

        Args:
            x_max: Maximum X value

        Returns:
            Growth analysis
        """
        x_values = [10 ** k for k in range(1, int(np.log10(x_max)) + 1)]
        return growth_rate_analysis(self.phi, x_values, self.precision)

    def prime_density_comparison(self, X: int) -> Dict[str, Any]:
        """
        Compare prime densities between classical and twisted.

        Args:
            X: Upper limit

        Returns:
            Density comparison
        """
        classical = self.get_classical_primes(X)
        twisted = self.get_twisted_primes(X)

        # Prime counting function comparison
        intervals = [X // 10 * i for i in range(1, 11)]

        classical_counts = [len([p for p in classical if p <= x]) for x in intervals]
        twisted_counts = [len([p for p in twisted if p <= x]) for x in intervals]

        # Compute density ratios
        density_ratios = [t / c if c > 0 else float('inf')
                         for t, c in zip(twisted_counts, classical_counts)]

        return {
            'X': X,
            'intervals': intervals,
            'classical_counts': classical_counts,
            'twisted_counts': twisted_counts,
            'density_ratios': density_ratios,
            'final_density_ratio': density_ratios[-1] if density_ratios else None,
            'interpretation': self._interpret_density_ratio(density_ratios[-1] if density_ratios else 1.0),
        }

    def _interpret_density_ratio(self, ratio: float) -> str:
        """Interpret the density ratio."""
        if ratio < 0.9:
            return f"Twisted primes are {(1-ratio)*100:.1f}% less dense than classical primes"
        elif ratio > 1.1:
            return f"Twisted primes are {(ratio-1)*100:.1f}% more dense than classical primes"
        else:
            return "Twisted and classical primes have similar density"

    def summary(self) -> str:
        """Generate a text summary of distribution analysis."""
        X = 10000
        analysis = self.full_analysis(X)

        lines = [
            f"Distribution Analysis for {self.phi.name}",
            "=" * 50,
            f"At X = {X}:",
            f"  Classical primes: {analysis['set_comparison']['n_classical']}",
            f"  Twisted primes: {analysis['set_comparison']['n_twisted']}",
            f"  Jaccard similarity: {analysis['set_comparison']['jaccard_similarity']:.4f}",
            f"  Distribution defect Δ_φ(X): {analysis['delta_analysis']['delta_phi_X']:.4f}",
            f"    - Log ratio component: {analysis['delta_analysis']['log_ratio_component']:.4f}",
            f"    - Symmetric difference: {analysis['delta_analysis']['symmetric_difference_component']}",
        ]

        return "\n".join(lines)

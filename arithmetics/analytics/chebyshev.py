"""
Twisted Chebyshev function and explicit formula implementation.

The twisted Chebyshev function is:
    ψ_φ(x) = Σ_{n≤x} Λ(n) φ(n)^{-1}

where Λ(n) is the von Mangoldt function.

The explicit formula relates ψ_φ(x) to the zeros of ζ_φ(s).
"""

from typing import List, Tuple, Optional, Union, Dict, Any
from mpmath import mp, mpf, mpc, log as mp_log, sqrt as mp_sqrt, fabs, floor as mp_floor
import numpy as np

from arithmetics.core.transfer import ArithmeticTransfer
from arithmetics.zeta.euler_product import sieve_primes


def von_mangoldt(n: int, prime_list: Optional[List[int]] = None) -> mpf:
    """
    Compute the von Mangoldt function Λ(n).

    Λ(n) = log(p) if n = p^k for some prime p and k ≥ 1
    Λ(n) = 0 otherwise

    Args:
        n: Positive integer
        prime_list: Optional list of primes (for efficiency)

    Returns:
        Λ(n) as mpf
    """
    if n <= 1:
        return mpf(0)

    # Check if n is a prime power
    # First, find if n = p^k for some prime p

    if prime_list is None:
        prime_list = sieve_primes(int(np.sqrt(n)) + 1)

    for p in prime_list:
        if p > n:
            break

        if n == p:
            return mp_log(mpf(p))

        # Check if n is a power of p
        m = n
        while m % p == 0:
            m //= p
            if m == 1:
                return mp_log(mpf(p))

        if m != n // p ** 0:  # n was not divisible by p
            continue

    # Check if n itself is prime (not in our limited prime_list)
    # Simple primality check for n
    if is_prime_simple(n):
        return mp_log(mpf(n))

    return mpf(0)


def is_prime_simple(n: int) -> bool:
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def precompute_von_mangoldt(limit: int) -> Dict[int, mpf]:
    """
    Precompute von Mangoldt function values up to limit.

    Args:
        limit: Upper bound

    Returns:
        Dictionary mapping n to Λ(n)
    """
    primes = sieve_primes(limit)
    prime_set = set(primes)

    lambda_values = {}

    for n in range(1, limit + 1):
        if n in prime_set:
            lambda_values[n] = mp_log(mpf(n))
        else:
            # Check if prime power
            val = mpf(0)
            for p in primes:
                if p > n:
                    break
                # Check if n = p^k
                m = n
                k = 0
                while m % p == 0:
                    m //= p
                    k += 1
                if m == 1 and k >= 1:
                    val = mp_log(mpf(p))
                    break
            lambda_values[n] = val

    return lambda_values


def psi_phi(
    x: Union[int, float, mpf],
    phi: ArithmeticTransfer,
    precision: int = 30,
    lambda_cache: Optional[Dict[int, mpf]] = None
) -> mpf:
    """
    Compute the twisted Chebyshev function ψ_φ(x).

    ψ_φ(x) = Σ_{n≤x} Λ(n) φ(n)^{-1}

    Args:
        x: Upper limit of sum
        phi: The arithmetic transfer
        precision: Decimal places
        lambda_cache: Pre-computed von Mangoldt values

    Returns:
        ψ_φ(x)
    """
    mp.dps = precision
    phi.precision = precision

    x_int = int(mp_floor(mpf(x)))

    if x_int < 2:
        return mpf(0)

    # Precompute von Mangoldt if not provided
    if lambda_cache is None:
        lambda_cache = precompute_von_mangoldt(x_int)

    total = mpf(0)

    for n in range(2, x_int + 1):
        lambda_n = lambda_cache.get(n, mpf(0))

        if lambda_n == 0:
            continue

        phi_n = phi.phi(n)
        if phi_n <= 0:
            continue

        total += lambda_n / phi_n

    return total


def psi_phi_classical(x: Union[int, float]) -> mpf:
    """
    Compute the classical Chebyshev function ψ(x) = Σ_{n≤x} Λ(n).

    This is the case φ(n) = 1 (identity for multiplication gives constant 1).

    Args:
        x: Upper limit

    Returns:
        ψ(x)
    """
    x_int = int(x)
    if x_int < 2:
        return mpf(0)

    lambda_cache = precompute_von_mangoldt(x_int)

    total = mpf(0)
    for n in range(2, x_int + 1):
        total += lambda_cache.get(n, mpf(0))

    return total


def explicit_formula(
    x: Union[int, float, mpf],
    phi: ArithmeticTransfer,
    zeros: List[complex],
    precision: int = 30
) -> Tuple[mpf, mpf]:
    """
    Evaluate the explicit formula for ψ_φ(x).

    ψ_φ(x) = -ζ'_φ(0)/ζ_φ(0) - Σ_ρ (x^ρ/ρ) φ^{-ρ} + R(x)

    This is a simplified version; the full formula involves more terms.

    Args:
        x: Point of evaluation
        phi: The arithmetic transfer
        zeros: List of known zeros of ζ_φ(s)
        precision: Decimal places

    Returns:
        Tuple (explicit_formula_value, sum_over_zeros)
    """
    mp.dps = precision
    phi.precision = precision

    x = mpf(x)

    # Sum over zeros: -Σ_ρ x^ρ / ρ
    zero_sum = mpc(0)

    for rho in zeros:
        rho_c = mpc(rho.real, rho.imag)

        if fabs(rho_c) < mpf(10) ** (-(precision - 5)):
            continue

        # x^ρ = exp(ρ log(x))
        x_rho = mp_log(x) * rho_c
        x_power_rho = mpc(np.cos(float(x_rho.imag)), np.sin(float(x_rho.imag)))
        x_power_rho *= mpf(np.exp(float(x_rho.real)))

        term = x_power_rho / rho_c
        zero_sum -= term

    # Take real part (zeros come in conjugate pairs)
    explicit_value = zero_sum.real

    return explicit_value, zero_sum.real


def explicit_formula_with_remainder(
    x: Union[int, float, mpf],
    phi: ArithmeticTransfer,
    zeros: List[complex],
    sigma_0: float = -0.5,
    precision: int = 30
) -> Dict[str, Any]:
    """
    Evaluate explicit formula and estimate remainder.

    Args:
        x: Point of evaluation
        phi: The arithmetic transfer
        zeros: List of zeros
        sigma_0: Abscissa of zero-free region estimate
        precision: Decimal places

    Returns:
        Dictionary with formula components
    """
    mp.dps = precision
    phi.precision = precision

    x = mpf(x)

    # Compute actual ψ_φ(x)
    psi_actual = psi_phi(x, phi, precision)

    # Compute explicit formula value
    explicit_val, zero_sum = explicit_formula(x, phi, zeros, precision)

    # Remainder
    remainder = psi_actual - explicit_val

    # Predicted remainder bound: O(x^{σ_0 + ε})
    remainder_bound = float(x) ** sigma_0

    return {
        'x': float(x),
        'psi_phi_x': float(psi_actual),
        'explicit_formula_value': float(explicit_val),
        'sum_over_zeros': float(zero_sum),
        'remainder_observed': float(remainder),
        'remainder_bound': remainder_bound,
        'n_zeros_used': len(zeros),
        'sigma_0': sigma_0,
    }


def psi_phi_grid(
    phi: ArithmeticTransfer,
    x_values: List[Union[int, float]],
    precision: int = 30
) -> List[Tuple[float, float]]:
    """
    Evaluate ψ_φ(x) on a grid of x values.

    Args:
        phi: The arithmetic transfer
        x_values: Points to evaluate
        precision: Decimal places

    Returns:
        List of (x, ψ_φ(x)) pairs
    """
    mp.dps = precision
    phi.precision = precision

    # Precompute von Mangoldt for efficiency
    max_x = int(max(x_values))
    lambda_cache = precompute_von_mangoldt(max_x)

    results = []
    for x in x_values:
        psi = psi_phi(x, phi, precision, lambda_cache)
        results.append((float(x), float(psi)))

    return results


def prime_counting_twisted(
    x: Union[int, float],
    phi: ArithmeticTransfer,
    precision: int = 30
) -> int:
    """
    Count φ-weighted primes up to x.

    π_φ(x) = #{p ≤ x : p is prime, weighted by 1/φ(p)}

    This is a simplified counting; for exact counting use Euler product primes.

    Args:
        x: Upper limit
        phi: The arithmetic transfer
        precision: Decimal places

    Returns:
        Weighted prime count
    """
    mp.dps = precision
    phi.precision = precision

    x_int = int(x)
    primes = sieve_primes(x_int)

    total = mpf(0)
    for p in primes:
        phi_p = phi.phi(p)
        if phi_p > 0:
            total += 1 / phi_p

    return float(total)


def estimate_sigma_0(
    phi: ArithmeticTransfer,
    zeros: List[complex],
    x_test_points: List[float] = None,
    precision: int = 30
) -> float:
    """
    Estimate the abscissa σ_0 of the zero-free region.

    We estimate σ_0 by finding the value that best explains the observed
    remainder in the explicit formula.

    Args:
        phi: The arithmetic transfer
        zeros: Known zeros
        x_test_points: Points to test (default: logarithmic scale 10 to 10^5)
        precision: Decimal places

    Returns:
        Estimated σ_0
    """
    if x_test_points is None:
        x_test_points = [10 ** k for k in range(1, 6)]

    mp.dps = precision
    phi.precision = precision

    # Compute remainders at test points
    remainders = []
    for x in x_test_points:
        result = explicit_formula_with_remainder(x, phi, zeros, precision=precision)
        if result['remainder_observed'] != 0:
            remainders.append((x, abs(result['remainder_observed'])))

    if len(remainders) < 2:
        return 0.0  # Not enough data

    # Fit R(x) ≈ C * x^σ_0
    # log(R) ≈ log(C) + σ_0 * log(x)
    log_x = np.array([np.log(x) for x, r in remainders])
    log_r = np.array([np.log(r) for x, r in remainders])

    # Linear regression
    if len(log_x) >= 2:
        slope, intercept = np.polyfit(log_x, log_r, 1)
        sigma_0_estimate = slope
    else:
        sigma_0_estimate = 0.0

    return sigma_0_estimate


class ChebyshevAnalyzer:
    """
    Comprehensive analyzer for twisted Chebyshev functions.
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
        self._lambda_cache = None
        self._psi_cache = {}

        mp.dps = precision
        phi.precision = precision

    def precompute(self, limit: int):
        """Precompute von Mangoldt values up to limit."""
        self._lambda_cache = precompute_von_mangoldt(limit)

    def psi(self, x: Union[int, float]) -> float:
        """Evaluate ψ_φ(x) with caching."""
        x_int = int(x)

        if x_int in self._psi_cache:
            return self._psi_cache[x_int]

        if self._lambda_cache is None or max(self._lambda_cache.keys(), default=0) < x_int:
            self.precompute(x_int)

        result = float(psi_phi(x, self.phi, self.precision, self._lambda_cache))
        self._psi_cache[x_int] = result

        return result

    def evaluate_grid(
        self,
        x_min: float = 10,
        x_max: float = 1000000,
        n_points: int = 50,
        log_scale: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Evaluate ψ_φ on a grid.

        Args:
            x_min: Minimum x
            x_max: Maximum x
            n_points: Number of points
            log_scale: Use logarithmic spacing

        Returns:
            List of (x, ψ_φ(x)) pairs
        """
        if log_scale:
            x_values = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
        else:
            x_values = np.linspace(x_min, x_max, n_points)

        return [(x, self.psi(x)) for x in x_values]

    def explicit_formula_analysis(
        self,
        zeros: List[complex],
        x_values: List[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze explicit formula at multiple points.

        Args:
            zeros: Known zeros of ζ_φ
            x_values: Points to analyze

        Returns:
            List of analysis dictionaries
        """
        if x_values is None:
            x_values = [10 ** k for k in range(1, 7)]

        results = []
        for x in x_values:
            result = explicit_formula_with_remainder(
                x, self.phi, zeros, precision=self.precision
            )
            results.append(result)

        return results

    def estimate_zero_free_region(self, zeros: List[complex]) -> Dict[str, Any]:
        """
        Estimate the zero-free region based on explicit formula.

        Args:
            zeros: Known zeros

        Returns:
            Dictionary with estimates
        """
        sigma_0 = estimate_sigma_0(self.phi, zeros, precision=self.precision)

        return {
            'sigma_0_estimate': sigma_0,
            'interpretation': f"ζ_φ(s) has no zeros for Re(s) ≥ {sigma_0:.3f}" if sigma_0 < 0 else "Unable to establish zero-free region",
            'n_zeros_used': len(zeros),
        }

    def growth_comparison(
        self,
        x_values: List[float] = None
    ) -> List[Dict[str, float]]:
        """
        Compare ψ_φ(x) with classical ψ(x) and x.

        Args:
            x_values: Points to compare

        Returns:
            List of comparison dictionaries
        """
        if x_values is None:
            x_values = [10 ** k for k in range(1, 6)]

        results = []
        for x in x_values:
            psi_twisted = self.psi(x)
            psi_classical = float(psi_phi_classical(x))

            results.append({
                'x': x,
                'psi_phi': psi_twisted,
                'psi_classical': psi_classical,
                'ratio': psi_twisted / psi_classical if psi_classical != 0 else float('inf'),
                'relative_to_x': psi_twisted / x if x != 0 else 0,
                'classical_relative_to_x': psi_classical / x if x != 0 else 0,
            })

        return results

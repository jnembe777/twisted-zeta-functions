"""
Dirichlet series evaluation for twisted zeta functions.

Computes ζ_φ(s) = Σ_{n=1}^∞ φ(n)^{-s} / n^s with high precision,
adaptive truncation, and acceleration techniques.
"""

from typing import Tuple, Optional, Union, Callable, List
from mpmath import (
    mp, mpf, mpc, power as mp_power, log as mp_log, exp as mp_exp,
    fabs, sqrt as mp_sqrt, pi as mp_pi, gamma as mp_gamma, zeta as mp_zeta,
    cos as mp_cos, sin as mp_sin
)
import numpy as np

from arithmetics.core.transfer import ArithmeticTransfer, ExponentialTransfer


def zeta_phi(
    s: Union[complex, mpc],
    phi: ArithmeticTransfer,
    n_max: int = 10000000,
    precision: int = 30,
    adaptive: bool = True,
    convergence_threshold: Optional[mpf] = None
) -> mpc:
    """
    Evaluate the twisted zeta function ζ_φ(s) via Dirichlet series.

    ζ_φ(s) = Σ_{n=1}^{N} φ(n)^{-s} / n^s

    For exponential transfer φ_α(n) = α^n:
    ζ_α(s) = Σ_{n=1}^{N} α^{-ns} / n^s

    Args:
        s: Complex argument (Re(s) > σ_c for convergence)
        phi: The arithmetic transfer function
        n_max: Maximum number of terms
        precision: Decimal places for mpmath
        adaptive: Use adaptive truncation based on convergence
        convergence_threshold: Stop when |term| < threshold

    Returns:
        The value ζ_φ(s) as a complex mpf
    """
    mp.dps = precision
    phi.precision = precision

    # Convert s to mpc
    if isinstance(s, complex):
        s = mpc(s.real, s.imag)
    elif not isinstance(s, mpc):
        s = mpc(s)

    if convergence_threshold is None:
        convergence_threshold = mpf(10) ** (-(precision - 5))

    # Initialize sum
    total = mpc(0)
    prev_total = mpc(0)

    # Track convergence
    convergence_count = 0
    required_convergence = 10  # Require stable for this many terms

    for n in range(1, n_max + 1):
        # Compute term: φ(n)^{-s} / n^s
        phi_n = phi.phi(n)

        if phi_n <= 0:
            # Skip invalid terms (shouldn't happen for valid transfers)
            continue

        # φ(n)^{-s} = exp(-s * log(φ(n)))
        log_phi_n = mp_log(phi_n)
        phi_term = mp_exp(-s * log_phi_n)

        # n^{-s} = exp(-s * log(n))
        n_term = mp_exp(-s * mp_log(mpf(n)))

        term = phi_term * n_term
        total += term

        # Check convergence
        if adaptive and n > 100:
            term_mag = fabs(term)
            if term_mag < convergence_threshold:
                convergence_count += 1
                if convergence_count >= required_convergence:
                    break
            else:
                convergence_count = 0

    return total


def zeta_phi_derivative(
    s: Union[complex, mpc],
    phi: ArithmeticTransfer,
    n_max: int = 10000000,
    precision: int = 30
) -> mpc:
    """
    Compute the derivative ζ'_φ(s) = dζ_φ/ds.

    d/ds[φ(n)^{-s}/n^s] = -log(φ(n)·n) · φ(n)^{-s}/n^s

    Args:
        s: Complex argument
        phi: The arithmetic transfer function
        n_max: Maximum number of terms
        precision: Decimal places

    Returns:
        ζ'_φ(s)
    """
    mp.dps = precision
    phi.precision = precision

    if isinstance(s, complex):
        s = mpc(s.real, s.imag)
    elif not isinstance(s, mpc):
        s = mpc(s)

    convergence_threshold = mpf(10) ** (-(precision - 5))

    total = mpc(0)

    for n in range(1, n_max + 1):
        phi_n = phi.phi(n)
        if phi_n <= 0:
            continue

        log_phi_n = mp_log(phi_n)
        log_n = mp_log(mpf(n))

        # Base term: φ(n)^{-s}/n^s
        base_term = mp_exp(-s * (log_phi_n + log_n))

        # Derivative term: multiply by -log(φ(n)·n)
        deriv_factor = -(log_phi_n + log_n)
        term = deriv_factor * base_term

        total += term

        # Check convergence
        if n > 100 and fabs(term) < convergence_threshold:
            break

    return total


def zeta_phi_accelerated(
    s: Union[complex, mpc],
    phi: ArithmeticTransfer,
    n_max: int = 1000000,
    precision: int = 30,
    method: str = 'euler_maclaurin'
) -> mpc:
    """
    Accelerated evaluation of ζ_φ(s) using Euler-Maclaurin or Richardson extrapolation.

    Args:
        s: Complex argument
        phi: The arithmetic transfer function
        n_max: Number of terms
        precision: Decimal places
        method: 'euler_maclaurin' or 'richardson'

    Returns:
        Accelerated estimate of ζ_φ(s)
    """
    mp.dps = precision
    phi.precision = precision

    if isinstance(s, complex):
        s = mpc(s.real, s.imag)

    if method == 'richardson':
        return _richardson_extrapolation(s, phi, n_max, precision)
    else:
        return _euler_maclaurin(s, phi, n_max, precision)


def _richardson_extrapolation(
    s: mpc,
    phi: ArithmeticTransfer,
    n_max: int,
    precision: int
) -> mpc:
    """
    Richardson extrapolation for series acceleration.

    Computes partial sums S_N, S_{2N}, S_{4N}, ... and extrapolates.
    """
    mp.dps = precision

    # Compute partial sums at different truncations
    ns = [n_max // 8, n_max // 4, n_max // 2, n_max]
    sums = []

    for n in ns:
        s_n = zeta_phi(s, phi, n, precision, adaptive=False)
        sums.append(s_n)

    # Richardson extrapolation formula (assuming O(1/N) convergence)
    # R_1 = (4*S_2N - S_N) / 3
    # R_2 = (4*R_1(2N) - R_1(N)) / 3, etc.
    if len(sums) >= 2:
        r1 = (4 * sums[-1] - sums[-2]) / 3
        return r1

    return sums[-1]


def _euler_maclaurin(
    s: mpc,
    phi: ArithmeticTransfer,
    n_max: int,
    precision: int
) -> mpc:
    """
    Euler-Maclaurin summation for series acceleration.

    Adds correction terms based on derivatives at endpoints.
    """
    mp.dps = precision

    # Basic sum
    total = zeta_phi(s, phi, n_max, precision, adaptive=False)

    # Euler-Maclaurin correction: add integral approximation for tail
    # For φ_α(n) = α^n, the tail integral can be estimated

    if isinstance(phi, ExponentialTransfer):
        alpha = phi.alpha
        # Approximate tail: ∫_{N}^∞ α^{-ns}/n^s dn
        # This is difficult in general; use asymptotic estimate
        n = mpf(n_max)
        log_alpha = mp_log(alpha)

        if s.real > 1:
            # Rough tail estimate
            tail_estimate = mp_exp(-s * (n * log_alpha + mp_log(n))) / (s - 1)
            total += tail_estimate

    return total


def zeta_phi_grid(
    phi: ArithmeticTransfer,
    re_range: Tuple[float, float] = (-1.0, 2.0),
    im_range: Tuple[float, float] = (0.0, 100.0),
    re_step: float = 0.1,
    im_step: float = 1.0,
    n_max: int = 100000,
    precision: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate ζ_φ(s) on a grid in the complex plane.

    Args:
        phi: The arithmetic transfer
        re_range: Range for Re(s)
        im_range: Range for Im(s)
        re_step: Step size for Re(s)
        im_step: Step size for Im(s)
        n_max: Maximum terms per evaluation
        precision: Decimal places

    Returns:
        Tuple (re_grid, im_grid, zeta_values) where zeta_values is complex array
    """
    mp.dps = precision
    phi.precision = precision

    re_vals = np.arange(re_range[0], re_range[1] + re_step, re_step)
    im_vals = np.arange(im_range[0], im_range[1] + im_step, im_step)

    re_grid, im_grid = np.meshgrid(re_vals, im_vals)
    zeta_values = np.zeros_like(re_grid, dtype=complex)

    for i, im_val in enumerate(im_vals):
        for j, re_val in enumerate(re_vals):
            s = mpc(re_val, im_val)
            z = zeta_phi(s, phi, n_max, precision)
            zeta_values[i, j] = complex(float(z.real), float(z.imag))

    return re_grid, im_grid, zeta_values


def find_convergence_abscissa(
    phi: ArithmeticTransfer,
    precision: int = 20,
    tol: float = 0.01
) -> float:
    """
    Estimate the abscissa of convergence σ_c for ζ_φ(s).

    The series converges for Re(s) > σ_c.

    Args:
        phi: The arithmetic transfer
        precision: Decimal places
        tol: Tolerance for bisection

    Returns:
        Estimated σ_c
    """
    mp.dps = precision
    phi.precision = precision

    # Test convergence at various σ values
    def test_convergence(sigma: float) -> bool:
        s = mpc(sigma, 10.0)  # Test at t=10
        try:
            z = zeta_phi(s, phi, n_max=10000, precision=precision, adaptive=True)
            return z.real == z.real  # Not NaN
        except (OverflowError, ValueError):
            return False

    # Binary search for σ_c
    sigma_low = -2.0
    sigma_high = 3.0

    # First, ensure bounds are valid
    while not test_convergence(sigma_high):
        sigma_high += 1.0
        if sigma_high > 10:
            return float('inf')

    # Binary search
    while sigma_high - sigma_low > tol:
        sigma_mid = (sigma_low + sigma_high) / 2
        if test_convergence(sigma_mid):
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    return sigma_high


def zeta_classical_check(
    s: Union[complex, mpc],
    precision: int = 30
) -> Tuple[mpc, mpc, mpf]:
    """
    Compare our identity transfer zeta with mpmath's built-in zeta.

    For identity transfer φ(n) = n:
    ζ_φ(s) = Σ φ(n)^{-s} / n^s = Σ n^{-s} / n^s = Σ n^{-2s} = ζ(2s)

    So we compare our ζ_φ(s) with mpmath's ζ(2s).

    Args:
        s: Complex argument
        precision: Decimal places

    Returns:
        Tuple (our_value, expected_value, relative_error)
    """
    from arithmetics.core.transfer import IdentityTransfer

    mp.dps = precision

    if isinstance(s, complex):
        s = mpc(s.real, s.imag)

    # Our implementation with identity transfer
    identity = IdentityTransfer()
    identity.precision = precision
    our_value = zeta_phi(s, identity, n_max=1000000, precision=precision)

    # For identity transfer, ζ_φ(s) = ζ(2s)
    mpmath_value = mp_zeta(2 * s)

    # Relative error
    if fabs(mpmath_value) > 0:
        rel_error = fabs(our_value - mpmath_value) / fabs(mpmath_value)
    else:
        rel_error = fabs(our_value - mpmath_value)

    return our_value, mpmath_value, rel_error


class DirichletSeriesEvaluator:
    """
    Class for efficient repeated evaluation of ζ_φ(s).

    Caches computed values and provides batch evaluation.
    """

    def __init__(
        self,
        phi: ArithmeticTransfer,
        n_max: int = 1000000,
        precision: int = 30
    ):
        """
        Initialize the evaluator.

        Args:
            phi: The arithmetic transfer
            n_max: Maximum terms for series
            precision: Decimal places
        """
        self.phi = phi
        self.n_max = n_max
        self.precision = precision
        self._cache = {}

        mp.dps = precision
        phi.precision = precision

        # Precompute phi values for efficiency
        self._phi_values = {}
        self._log_phi_values = {}
        self._log_n_values = {}

    def _precompute(self, n_terms: int):
        """Precompute values for first n_terms."""
        for n in range(1, n_terms + 1):
            if n not in self._phi_values:
                phi_n = self.phi.phi(n)
                self._phi_values[n] = phi_n
                if phi_n > 0:
                    self._log_phi_values[n] = mp_log(phi_n)
                self._log_n_values[n] = mp_log(mpf(n))

    def evaluate(self, s: Union[complex, mpc], use_cache: bool = True) -> mpc:
        """
        Evaluate ζ_φ(s).

        Args:
            s: Complex argument
            use_cache: Whether to cache/use cached results

        Returns:
            ζ_φ(s)
        """
        if isinstance(s, complex):
            key = (s.real, s.imag)
            s = mpc(s.real, s.imag)
        else:
            key = (float(s.real), float(s.imag))

        if use_cache and key in self._cache:
            return self._cache[key]

        result = zeta_phi(s, self.phi, self.n_max, self.precision)

        if use_cache:
            self._cache[key] = result

        return result

    def evaluate_derivative(self, s: Union[complex, mpc]) -> mpc:
        """Evaluate ζ'_φ(s)."""
        return zeta_phi_derivative(s, self.phi, self.n_max, self.precision)

    def batch_evaluate(
        self,
        s_values: List[Union[complex, mpc]]
    ) -> List[mpc]:
        """
        Evaluate ζ_φ(s) for multiple s values.

        Args:
            s_values: List of complex arguments

        Returns:
            List of ζ_φ(s) values
        """
        return [self.evaluate(s) for s in s_values]

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()

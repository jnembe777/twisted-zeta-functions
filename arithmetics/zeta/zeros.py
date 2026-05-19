"""
Zero finding module for twisted zeta functions.

Implements Newton-Raphson in the complex plane with grid search initialization
and argument principle verification for robust zero detection.
"""

from typing import List, Tuple, Optional, Union, Callable, NamedTuple
from dataclasses import dataclass
from mpmath import mp, mpf, mpc, fabs, sqrt as mp_sqrt, pi as mp_pi, log as mp_log
import numpy as np

from arithmetics.core.transfer import ArithmeticTransfer
from arithmetics.zeta.dirichlet import zeta_phi, zeta_phi_derivative, DirichletSeriesEvaluator


@dataclass
class ZeroSearchRegion:
    """Definition of a search region in the complex plane."""
    re_min: float
    re_max: float
    im_min: float
    im_max: float
    re_step: float = 0.1
    im_step: float = 0.5

    def grid_points(self) -> List[complex]:
        """Generate grid points within this region."""
        points = []
        re_vals = np.arange(self.re_min, self.re_max + self.re_step, self.re_step)
        im_vals = np.arange(self.im_min, self.im_max + self.im_step, self.im_step)

        for re in re_vals:
            for im in im_vals:
                points.append(complex(re, im))

        return points


class ZeroResult(NamedTuple):
    """Result of a zero search."""
    zero: complex
    error_estimate: float
    n_iterations: int
    method: str
    verified: bool
    multiplicity: int = 1


def newton_raphson_complex(
    f: Callable[[mpc], mpc],
    f_prime: Callable[[mpc], mpc],
    z0: Union[complex, mpc],
    tol: float = 1e-20,
    max_iter: int = 100,
    precision: int = 30
) -> Tuple[Optional[mpc], int, float]:
    """
    Newton-Raphson iteration in the complex plane.

    z_{n+1} = z_n - f(z_n) / f'(z_n)

    Args:
        f: The function whose zero we seek
        f_prime: Derivative of f
        z0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum iterations
        precision: Decimal places

    Returns:
        Tuple (zero, n_iterations, final_error) or (None, n_iterations, final_error) if failed
    """
    mp.dps = precision

    if isinstance(z0, complex):
        z = mpc(z0.real, z0.imag)
    else:
        z = z0

    tol_mpf = mpf(tol)

    for i in range(max_iter):
        fz = f(z)
        fpz = f_prime(z)

        # Check for division by zero
        if fabs(fpz) < mpf(10) ** (-(precision - 5)):
            # Derivative too small, likely at a critical point
            return None, i + 1, float(fabs(fz))

        # Newton step
        z_new = z - fz / fpz

        # Check convergence
        delta = fabs(z_new - z)
        if delta < tol_mpf:
            # Also verify f(z) is small
            fz_new = f(z_new)
            if fabs(fz_new) < mpf(10) ** (-(precision // 2)):
                return z_new, i + 1, float(fabs(fz_new))

        z = z_new

    # Max iterations reached
    fz_final = f(z)
    return z, max_iter, float(fabs(fz_final))


def secant_method_complex(
    f: Callable[[mpc], mpc],
    z0: Union[complex, mpc],
    z1: Union[complex, mpc],
    tol: float = 1e-20,
    max_iter: int = 100,
    precision: int = 30
) -> Tuple[Optional[mpc], int, float]:
    """
    Secant method in the complex plane (no derivative required).

    Args:
        f: The function whose zero we seek
        z0, z1: Two initial guesses
        tol: Convergence tolerance
        max_iter: Maximum iterations
        precision: Decimal places

    Returns:
        Tuple (zero, n_iterations, final_error)
    """
    mp.dps = precision

    if isinstance(z0, complex):
        z0 = mpc(z0.real, z0.imag)
    if isinstance(z1, complex):
        z1 = mpc(z1.real, z1.imag)

    tol_mpf = mpf(tol)

    f0 = f(z0)
    f1 = f(z1)

    for i in range(max_iter):
        denom = f1 - f0
        if fabs(denom) < mpf(10) ** (-(precision - 5)):
            return None, i + 1, float(fabs(f1))

        z2 = z1 - f1 * (z1 - z0) / denom

        delta = fabs(z2 - z1)
        if delta < tol_mpf:
            f2 = f(z2)
            if fabs(f2) < mpf(10) ** (-(precision // 2)):
                return z2, i + 1, float(fabs(f2))

        z0, z1 = z1, z2
        f0, f1 = f1, f(z2)

    return z1, max_iter, float(fabs(f1))


def argument_principle_count(
    f: Callable[[mpc], mpc],
    center: complex,
    radius: float,
    n_points: int = 100,
    precision: int = 30
) -> int:
    """
    Count zeros inside a circle using the argument principle.

    N = (1/2πi) ∮ f'(z)/f(z) dz = number of zeros inside contour

    We approximate this by tracking the winding number of f(z) around the origin
    as z traverses the circle.

    Args:
        f: The function
        center: Center of the circle
        radius: Radius of the circle
        n_points: Number of points for contour integration
        precision: Decimal places

    Returns:
        Estimated number of zeros inside the circle
    """
    mp.dps = precision

    # Parameterize circle: z(t) = center + radius * exp(2πit), t ∈ [0, 1]
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Track total argument change
    total_arg_change = 0.0
    prev_f = None

    for theta in angles:
        z = mpc(center.real + radius * np.cos(theta),
                center.imag + radius * np.sin(theta))
        fz = f(z)

        if fabs(fz) < mpf(10) ** (-(precision - 5)):
            # Hit a zero on the contour
            continue

        if prev_f is not None:
            # Compute argument change
            # arg(f(z)) - arg(f(z_prev))
            arg_current = float(mp_log(fz).imag)
            arg_prev = float(mp_log(prev_f).imag)

            delta_arg = arg_current - arg_prev

            # Handle branch cut: if jump > π, subtract 2π; if < -π, add 2π
            if delta_arg > np.pi:
                delta_arg -= 2 * np.pi
            elif delta_arg < -np.pi:
                delta_arg += 2 * np.pi

            total_arg_change += delta_arg

        prev_f = fz

    # Close the contour
    if prev_f is not None:
        z = mpc(center.real + radius, center.imag)
        fz = f(z)
        if fabs(fz) >= mpf(10) ** (-(precision - 5)):
            arg_current = float(mp_log(fz).imag)
            arg_prev = float(mp_log(prev_f).imag)
            delta_arg = arg_current - arg_prev
            if delta_arg > np.pi:
                delta_arg -= 2 * np.pi
            elif delta_arg < -np.pi:
                delta_arg += 2 * np.pi
            total_arg_change += delta_arg

    # Number of zeros = total_arg_change / (2π)
    n_zeros = round(total_arg_change / (2 * np.pi))

    return max(0, n_zeros)


def verify_zero(
    f: Callable[[mpc], mpc],
    zero: complex,
    radius: float = 0.1,
    precision: int = 30
) -> Tuple[bool, int]:
    """
    Verify that a zero exists and estimate its multiplicity.

    Args:
        f: The function
        zero: Candidate zero
        radius: Radius for argument principle test
        precision: Decimal places

    Returns:
        Tuple (is_zero, multiplicity)
    """
    mp.dps = precision

    # First check that f(zero) is small
    z = mpc(zero.real, zero.imag)
    fz = f(z)

    if fabs(fz) > mpf(10) ** (-(precision // 3)):
        return False, 0

    # Count zeros in neighborhood using argument principle
    count = argument_principle_count(f, zero, radius, precision=precision)

    return count > 0, count


def grid_search_zeros(
    f: Callable[[mpc], mpc],
    region: ZeroSearchRegion,
    threshold: float = 0.1,
    precision: int = 20
) -> List[complex]:
    """
    Find candidate zeros by grid search.

    Identifies points where |f(z)| is locally minimal.

    Args:
        f: The function
        region: Search region
        threshold: Maximum |f(z)| to consider as candidate
        precision: Decimal places

    Returns:
        List of candidate zero locations
    """
    mp.dps = precision

    candidates = []
    points = region.grid_points()

    # Evaluate f on grid
    values = {}
    for p in points:
        z = mpc(p.real, p.imag)
        fz = f(z)
        values[p] = float(fabs(fz))

    # Find local minima
    for p in points:
        val = values[p]

        if val > threshold:
            continue

        # Check if local minimum
        is_local_min = True
        for dp_re in [-region.re_step, 0, region.re_step]:
            for dp_im in [-region.im_step, 0, region.im_step]:
                if dp_re == 0 and dp_im == 0:
                    continue
                neighbor = complex(p.real + dp_re, p.imag + dp_im)
                if neighbor in values and values[neighbor] < val:
                    is_local_min = False
                    break
            if not is_local_min:
                break

        if is_local_min:
            candidates.append(p)

    return candidates


def find_zeros(
    phi: ArithmeticTransfer,
    region: ZeroSearchRegion,
    tol: float = 1e-20,
    max_iter: int = 100,
    precision: int = 20,
    n_max: int = 100000,
    verify: bool = True
) -> List[ZeroResult]:
    """
    Find zeros of ζ_φ(s) in a given region.

    Uses grid search for initialization followed by Newton-Raphson refinement.

    Args:
        phi: The arithmetic transfer
        region: Search region
        tol: Convergence tolerance
        max_iter: Maximum Newton iterations
        precision: Decimal places
        n_max: Maximum terms in zeta evaluation
        verify: Whether to verify zeros with argument principle

    Returns:
        List of ZeroResult objects
    """
    mp.dps = precision
    phi.precision = precision

    # Create evaluator
    evaluator = DirichletSeriesEvaluator(phi, n_max, precision)

    def f(z):
        return evaluator.evaluate(z, use_cache=True)

    def f_prime(z):
        return evaluator.evaluate_derivative(z)

    # Grid search for candidates
    candidates = grid_search_zeros(f, region, threshold=1.0, precision=precision)

    # Refine each candidate
    zeros = []
    found_zeros = set()  # Track to avoid duplicates

    for candidate in candidates:
        # Newton-Raphson refinement
        zero, n_iter, error = newton_raphson_complex(
            f, f_prime, candidate, tol, max_iter, precision
        )

        if zero is None:
            continue

        # Round for duplicate detection
        zero_key = (round(float(zero.real), 10), round(float(zero.imag), 10))

        if zero_key in found_zeros:
            continue

        # Verify if requested
        if verify:
            is_zero, mult = verify_zero(f, complex(float(zero.real), float(zero.imag)),
                                        radius=0.01, precision=precision)
            if not is_zero:
                continue
            verified = True
            multiplicity = mult
        else:
            verified = False
            multiplicity = 1

        found_zeros.add(zero_key)

        zeros.append(ZeroResult(
            zero=complex(float(zero.real), float(zero.imag)),
            error_estimate=error,
            n_iterations=n_iter,
            method='newton',
            verified=verified,
            multiplicity=multiplicity
        ))

    # Sort by imaginary part
    zeros.sort(key=lambda z: (z.zero.imag, z.zero.real))

    return zeros


def find_zeros_near_critical_line(
    phi: ArithmeticTransfer,
    t_max: float = 50.0,
    delta_re: float = 0.1,
    im_step: float = 0.5,
    precision: int = 20,
    n_max: int = 100000
) -> List[ZeroResult]:
    """
    Find zeros near the critical line Re(s) = 1/2.

    Args:
        phi: The arithmetic transfer
        t_max: Maximum imaginary part to search
        delta_re: Half-width around Re(s) = 0.5
        im_step: Step size for imaginary part
        precision: Decimal places
        n_max: Maximum terms in zeta

    Returns:
        List of zeros near critical line
    """
    region = ZeroSearchRegion(
        re_min=0.5 - delta_re,
        re_max=0.5 + delta_re,
        im_min=0.0,
        im_max=t_max,
        re_step=delta_re / 5,
        im_step=im_step
    )

    return find_zeros(phi, region, precision=precision, n_max=n_max)


def track_zero_trajectory(
    base_phi: Callable[[float], ArithmeticTransfer],
    alpha_range: Tuple[float, float],
    n_alpha: int = 20,
    initial_zero: complex = complex(0.5, 14.13),
    precision: int = 20,
    n_max: int = 50000
) -> List[Tuple[float, complex]]:
    """
    Track how a zero moves as the transfer parameter varies.

    Args:
        base_phi: Function that creates a transfer for given alpha
        alpha_range: Range of alpha values (min, max)
        n_alpha: Number of alpha values to sample
        initial_zero: Starting zero location (at alpha_min)
        precision: Decimal places
        n_max: Maximum terms in zeta

    Returns:
        List of (alpha, zero_location) pairs
    """
    mp.dps = precision

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    trajectory = []

    current_zero = initial_zero

    for alpha in alphas:
        phi = base_phi(alpha)
        phi.precision = precision

        evaluator = DirichletSeriesEvaluator(phi, n_max, precision)

        def f(z):
            return evaluator.evaluate(z)

        def f_prime(z):
            return evaluator.evaluate_derivative(z)

        # Refine from current position
        zero, n_iter, error = newton_raphson_complex(
            f, f_prime, current_zero, tol=1e-15, max_iter=50, precision=precision
        )

        if zero is not None:
            current_zero = complex(float(zero.real), float(zero.imag))
            trajectory.append((alpha, current_zero))

        evaluator.clear_cache()

    return trajectory


class ZeroFinder:
    """
    Comprehensive zero finder for twisted zeta functions.
    """

    def __init__(
        self,
        phi: ArithmeticTransfer,
        precision: int = 20,
        n_max: int = 100000
    ):
        """
        Initialize the zero finder.

        Args:
            phi: The arithmetic transfer
            precision: Decimal places
            n_max: Maximum terms in zeta evaluation
        """
        self.phi = phi
        self.precision = precision
        self.n_max = n_max
        self.evaluator = DirichletSeriesEvaluator(phi, n_max, precision)
        self._found_zeros = []

        mp.dps = precision
        phi.precision = precision

    def find_in_region(
        self,
        region: ZeroSearchRegion,
        tol: float = 1e-15
    ) -> List[ZeroResult]:
        """Find zeros in a specified region."""
        zeros = find_zeros(
            self.phi, region, tol=tol,
            precision=self.precision, n_max=self.n_max
        )
        self._found_zeros.extend(zeros)
        return zeros

    def find_near_critical_line(
        self,
        t_max: float = 50.0,
        delta_re: float = 0.1
    ) -> List[ZeroResult]:
        """Find zeros near Re(s) = 1/2."""
        zeros = find_zeros_near_critical_line(
            self.phi, t_max, delta_re,
            precision=self.precision, n_max=self.n_max
        )
        self._found_zeros.extend(zeros)
        return zeros

    def find_extended_region(self, t_max: float = 100.0) -> List[ZeroResult]:
        """Search extended region [-0.5, 1.5] x [0, t_max]."""
        region = ZeroSearchRegion(
            re_min=-0.5, re_max=1.5,
            im_min=0.0, im_max=t_max,
            re_step=0.05, im_step=1.0
        )
        return self.find_in_region(region)

    def all_zeros(self) -> List[ZeroResult]:
        """Return all found zeros, sorted by imaginary part."""
        # Remove duplicates
        unique = {}
        for z in self._found_zeros:
            key = (round(z.zero.real, 10), round(z.zero.imag, 10))
            if key not in unique or z.error_estimate < unique[key].error_estimate:
                unique[key] = z

        return sorted(unique.values(), key=lambda z: z.zero.imag)

    def zeros_on_critical_line(self, tolerance: float = 0.01) -> List[ZeroResult]:
        """Return zeros with Re(s) ≈ 0.5."""
        return [z for z in self.all_zeros()
                if abs(z.zero.real - 0.5) < tolerance]

    def zeros_off_critical_line(self, tolerance: float = 0.01) -> List[ZeroResult]:
        """Return zeros with Re(s) not close to 0.5."""
        return [z for z in self.all_zeros()
                if abs(z.zero.real - 0.5) >= tolerance]

    def statistics(self) -> dict:
        """Compute statistics about found zeros."""
        zeros = self.all_zeros()

        if not zeros:
            return {'n_zeros': 0}

        re_parts = [z.zero.real for z in zeros]
        im_parts = [z.zero.imag for z in zeros]

        on_critical = self.zeros_on_critical_line()
        off_critical = self.zeros_off_critical_line()

        return {
            'n_zeros': len(zeros),
            'n_on_critical_line': len(on_critical),
            'n_off_critical_line': len(off_critical),
            'fraction_on_critical': len(on_critical) / len(zeros) if zeros else 0,
            'mean_re': np.mean(re_parts),
            'std_re': np.std(re_parts),
            'min_re': min(re_parts),
            'max_re': max(re_parts),
            'mean_distance_to_critical': np.mean([abs(r - 0.5) for r in re_parts]),
            'im_range': (min(im_parts), max(im_parts)),
            'avg_error_estimate': np.mean([z.error_estimate for z in zeros]),
        }

    def clear(self):
        """Clear found zeros and cache."""
        self._found_zeros.clear()
        self.evaluator.clear_cache()

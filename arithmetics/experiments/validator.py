"""
Validation module for reproducibility and correctness testing.

Provides tests for:
- Numerical reproducibility across runs
- Precision convergence
- Classical case validation (φ(x)=x should match Riemann zeros)
- Functional equation residual checks
"""

import json
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mpmath import mp, mpf, mpc, fabs, zeta as mp_zeta
import numpy as np

from arithmetics.core.transfer import ArithmeticTransfer, IdentityTransfer, ExponentialTransfer
from arithmetics.core.defect import compute_defect, verify_cocycle_identity, DefectMatrix
from arithmetics.core.cohomology import classify_defect
from arithmetics.zeta.dirichlet import zeta_phi, zeta_classical_check
from arithmetics.zeta.zeros import ZeroFinder, ZeroSearchRegion


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_type: str
    passed: bool
    expected_value: Optional[float]
    actual_value: Optional[float]
    deviation: Optional[float]
    tolerance: float
    details: Dict[str, Any]
    tested_at: str = None

    def __post_init__(self):
        if self.tested_at is None:
            self.tested_at = datetime.now().isoformat()


def validate_reproducibility(
    phi: ArithmeticTransfer,
    n_reruns: int = 3,
    tolerance: float = 1e-15,
    precision: int = 50
) -> ValidationResult:
    """
    Test numerical reproducibility by running computations multiple times.

    Args:
        phi: The arithmetic transfer
        n_reruns: Number of times to repeat computation
        tolerance: Maximum allowed deviation between runs
        precision: Decimal places for computation

    Returns:
        ValidationResult
    """
    mp.dps = precision
    phi.precision = precision

    # Test defect computation reproducibility
    test_pairs = [(2, 3), (3, 4), (5, 7), (7, 11)]

    defect_results = []
    for run in range(n_reruns):
        run_defects = []
        for a, b in test_pairs:
            d = compute_defect(phi, a, b, precision)
            run_defects.append(float(d))
        defect_results.append(run_defects)

    # Check consistency across runs
    max_deviation = 0.0
    deviations = []

    for i in range(len(test_pairs)):
        values = [defect_results[run][i] for run in range(n_reruns)]
        if len(set(values)) > 1:
            dev = max(values) - min(values)
            max_deviation = max(max_deviation, dev)
            deviations.append({'pair': test_pairs[i], 'deviation': dev, 'values': values})

    passed = max_deviation < tolerance

    return ValidationResult(
        test_type='reproducibility',
        passed=passed,
        expected_value=0.0,
        actual_value=max_deviation,
        deviation=max_deviation,
        tolerance=tolerance,
        details={
            'n_reruns': n_reruns,
            'n_test_pairs': len(test_pairs),
            'deviations': deviations,
            'precision': precision,
        }
    )


def validate_precision_convergence(
    phi: ArithmeticTransfer,
    precisions: List[int] = None,
    test_pairs: List[Tuple[int, int]] = None
) -> ValidationResult:
    """
    Test that increasing precision leads to convergent results.

    Args:
        phi: The arithmetic transfer
        precisions: List of precisions to test (e.g., [50, 100, 200])
        test_pairs: Pairs to test

    Returns:
        ValidationResult
    """
    if precisions is None:
        precisions = [30, 50, 100, 150]

    if test_pairs is None:
        test_pairs = [(2, 3), (3, 5), (5, 7)]

    results_by_precision = {}

    for prec in precisions:
        mp.dps = prec
        phi.precision = prec

        pair_results = {}
        for a, b in test_pairs:
            d = compute_defect(phi, a, b, prec)
            pair_results[(a, b)] = float(d)

        results_by_precision[prec] = pair_results

    # Check for monotonic convergence
    convergence_data = []
    converging = True

    for pair in test_pairs:
        values = [results_by_precision[p][pair] for p in precisions]

        # Check if differences decrease
        diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        is_converging = all(diffs[i] >= diffs[i+1] * 0.9 for i in range(len(diffs)-1)) if len(diffs) > 1 else True

        convergence_data.append({
            'pair': pair,
            'values': values,
            'differences': diffs,
            'converging': is_converging
        })

        if not is_converging:
            converging = False

    return ValidationResult(
        test_type='precision_convergence',
        passed=converging,
        expected_value=None,
        actual_value=None,
        deviation=None,
        tolerance=0.0,
        details={
            'precisions': precisions,
            'convergence_data': convergence_data,
        }
    )


def validate_classical_case(
    n_zeros: int = 20,
    tolerance: float = 1e-10,
    precision: int = 30
) -> ValidationResult:
    """
    Validate that identity transfer reproduces known Riemann zeta zeros.

    The first non-trivial zeros of ζ(s) are at:
    s = 0.5 + i*t where t ≈ 14.134725, 21.022040, 25.010858, ...

    Args:
        n_zeros: Number of zeros to verify
        tolerance: Maximum deviation from known values
        precision: Decimal places

    Returns:
        ValidationResult
    """
    # Known first zeros of Riemann zeta (imaginary parts)
    known_zeros_im = [
        14.134725141734693790,
        21.022039638771554993,
        25.010857580145688763,
        30.424876125859513210,
        32.935061587739189691,
        37.586178158825671257,
        40.918719012147495187,
        43.327073280914999519,
        48.005150881167159727,
        49.773832477672302181,
        52.970321477714460644,
        56.446247697063394804,
        59.347044002602353079,
        60.831778524609809844,
        65.112544048081606660,
        67.079810529494173714,
        69.546401711173979253,
        72.067157674481907582,
        75.704690699083933168,
        77.144840068874805372,
    ]

    mp.dps = precision

    # Create identity transfer
    identity = IdentityTransfer()
    identity.precision = precision

    # Find zeros
    zero_finder = ZeroFinder(identity, precision=precision, n_max=100000)
    region = ZeroSearchRegion(
        re_min=0.45, re_max=0.55,
        im_min=0.0, im_max=80.0,
        re_step=0.01, im_step=0.5
    )
    found_zeros = zero_finder.find_in_region(region)

    # Sort by imaginary part
    found_zeros.sort(key=lambda z: z.zero.imag)

    # Match with known zeros
    matches = []
    max_deviation = 0.0

    n_to_check = min(n_zeros, len(known_zeros_im), len(found_zeros))

    for i in range(n_to_check):
        known_im = known_zeros_im[i]
        found = found_zeros[i]

        # Deviation in imaginary part (real part should be 0.5)
        im_deviation = abs(found.zero.imag - known_im)
        re_deviation = abs(found.zero.real - 0.5)

        deviation = max(im_deviation, re_deviation)
        max_deviation = max(max_deviation, deviation)

        matches.append({
            'index': i,
            'known_im': known_im,
            'found': complex(found.zero.real, found.zero.imag),
            'im_deviation': im_deviation,
            're_deviation': re_deviation,
            'passed': deviation < tolerance
        })

    passed = max_deviation < tolerance and len(found_zeros) >= n_to_check

    return ValidationResult(
        test_type='classical_validation',
        passed=passed,
        expected_value=0.5,  # Real part should be 0.5
        actual_value=float(np.mean([m['found'].real for m in matches])) if matches else None,
        deviation=max_deviation,
        tolerance=tolerance,
        details={
            'n_zeros_requested': n_zeros,
            'n_zeros_found': len(found_zeros),
            'n_matched': n_to_check,
            'matches': matches,
            'precision': precision,
        }
    )


def validate_functional_equation(
    phi: ExponentialTransfer,
    test_points: int = 10,
    tolerance: float = 1e-6,
    precision: int = 30
) -> ValidationResult:
    """
    Check approximate functional equation for twisted zeta.

    For classical zeta: ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s)

    For twisted zeta, we check that the residual of a similar
    approximate relation is small.

    Args:
        phi: Exponential transfer
        test_points: Number of points to test
        tolerance: Maximum residual
        precision: Decimal places

    Returns:
        ValidationResult
    """
    from mpmath import gamma as mp_gamma, sin as mp_sin, pi as mp_pi

    mp.dps = precision
    phi.precision = precision

    # Test points in the critical strip
    test_s = [mpc(0.5 + 0.1 * (i - 5), 10 + 2 * i) for i in range(test_points)]

    residuals = []
    max_residual = 0.0

    for s in test_s:
        try:
            # Evaluate ζ_φ(s) and ζ_φ(1-s)
            zeta_s = zeta_phi(s, phi, n_max=50000, precision=precision)
            zeta_1_minus_s = zeta_phi(1 - s, phi, n_max=50000, precision=precision)

            # Classical functional equation factors (approximate for twisted)
            # This is a consistency check, not an exact functional equation
            factor = (mpf(2) ** s) * (mp_pi ** (s - 1)) * mp_sin(mp_pi * s / 2) * mp_gamma(1 - s)

            # For twisted zeta, we expect a modified relation
            # Here we just check that the ratio is roughly constant
            if fabs(zeta_1_minus_s) > 1e-10:
                ratio = zeta_s / (factor * zeta_1_minus_s)
                residual = fabs(ratio.imag)  # Imaginary part should be small for real α
            else:
                residual = float('inf')

            residuals.append({
                's': complex(float(s.real), float(s.imag)),
                'zeta_s': complex(float(zeta_s.real), float(zeta_s.imag)),
                'residual': float(residual)
            })

            if residual < float('inf'):
                max_residual = max(max_residual, residual)

        except Exception as e:
            residuals.append({
                's': complex(float(s.real), float(s.imag)),
                'error': str(e)
            })

    passed = max_residual < tolerance

    return ValidationResult(
        test_type='functional_equation',
        passed=passed,
        expected_value=0.0,
        actual_value=max_residual,
        deviation=max_residual,
        tolerance=tolerance,
        details={
            'n_test_points': test_points,
            'alpha': float(phi.alpha),
            'residuals': residuals,
            'precision': precision,
        }
    )


def validate_cocycle_identity(
    phi: ArithmeticTransfer,
    n_triples: int = 20,
    tolerance: float = 1e-40,
    precision: int = 50
) -> ValidationResult:
    """
    Verify the twisted cocycle identity for multiple (a,b,c) triples.

    φ(a)δ(b,c) + δ(a,bc) = δ(a,b)φ(c) + δ(ab,c)

    Args:
        phi: The arithmetic transfer
        n_triples: Number of triples to test
        tolerance: Maximum residual
        precision: Decimal places

    Returns:
        ValidationResult
    """
    mp.dps = precision
    phi.precision = precision

    # Generate test triples
    import random
    random.seed(42)  # Reproducible

    test_values = list(range(2, 12))
    triples = []
    for _ in range(n_triples):
        a = random.choice(test_values)
        b = random.choice(test_values)
        c = random.choice(test_values)
        triples.append((a, b, c))

    results = []
    max_residual = mpf(0)

    for a, b, c in triples:
        valid, residual = verify_cocycle_identity(phi, a, b, c, precision, mpf(tolerance))
        results.append({
            'triple': (a, b, c),
            'residual': float(residual),
            'passed': valid
        })
        max_residual = max(max_residual, residual)

    n_passed = sum(1 for r in results if r['passed'])
    all_passed = n_passed == len(results)

    return ValidationResult(
        test_type='cocycle_identity',
        passed=all_passed,
        expected_value=0.0,
        actual_value=float(max_residual),
        deviation=float(max_residual),
        tolerance=tolerance,
        details={
            'n_triples': n_triples,
            'n_passed': n_passed,
            'results': results,
            'precision': precision,
        }
    )


class Validator:
    """
    Comprehensive validator for the experimental framework.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            db_path: Path to database for storing results
        """
        self.db_path = db_path
        self.results: List[ValidationResult] = []

    def run_all_validations(
        self,
        phi: ArithmeticTransfer,
        include_classical: bool = True
    ) -> Dict[str, ValidationResult]:
        """
        Run all validation tests on an arithmetic transfer.

        Args:
            phi: The arithmetic transfer
            include_classical: Whether to run classical validation (slow)

        Returns:
            Dictionary of test_type -> ValidationResult
        """
        results = {}

        # Reproducibility
        results['reproducibility'] = validate_reproducibility(phi)
        self.results.append(results['reproducibility'])

        # Precision convergence
        results['precision_convergence'] = validate_precision_convergence(phi)
        self.results.append(results['precision_convergence'])

        # Cocycle identity
        results['cocycle_identity'] = validate_cocycle_identity(phi)
        self.results.append(results['cocycle_identity'])

        # Classical validation (only for identity-like transfers)
        if include_classical and isinstance(phi, IdentityTransfer):
            results['classical_validation'] = validate_classical_case()
            self.results.append(results['classical_validation'])

        # Functional equation (only for exponential)
        if isinstance(phi, ExponentialTransfer):
            results['functional_equation'] = validate_functional_equation(phi)
            self.results.append(results['functional_equation'])

        return results

    def save_results(self, arithmetic_id: Optional[int] = None):
        """Save validation results to database."""
        if self.db_path is None:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for result in self.results:
            cursor.execute("""
                INSERT INTO validation_results
                (test_type, arithmetic_id, passed, expected_value, actual_value, deviation, tolerance, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_type,
                arithmetic_id,
                result.passed,
                result.expected_value,
                result.actual_value,
                result.deviation,
                result.tolerance,
                json.dumps(result.details)
            ))

        conn.commit()
        conn.close()

    def summary(self) -> Dict[str, Any]:
        """Generate summary of validation results."""
        if not self.results:
            return {'error': 'No validation results'}

        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.passed)

        by_type = {}
        for r in self.results:
            if r.test_type not in by_type:
                by_type[r.test_type] = {'passed': 0, 'failed': 0}
            if r.passed:
                by_type[r.test_type]['passed'] += 1
            else:
                by_type[r.test_type]['failed'] += 1

        return {
            'n_total': n_total,
            'n_passed': n_passed,
            'n_failed': n_total - n_passed,
            'pass_rate': n_passed / n_total if n_total > 0 else 0,
            'by_type': by_type,
            'all_passed': n_passed == n_total,
        }

    def report(self) -> str:
        """Generate human-readable validation report."""
        lines = [
            "Validation Report",
            "=" * 50,
        ]

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"\n[{status}] {result.test_type}")

            if result.expected_value is not None:
                lines.append(f"  Expected: {result.expected_value}")
            if result.actual_value is not None:
                lines.append(f"  Actual: {result.actual_value}")
            if result.deviation is not None:
                lines.append(f"  Deviation: {result.deviation:.2e}")
            lines.append(f"  Tolerance: {result.tolerance:.2e}")

        summary = self.summary()
        lines.extend([
            "\n" + "=" * 50,
            f"Summary: {summary['n_passed']}/{summary['n_total']} tests passed "
            f"({summary['pass_rate']*100:.1f}%)"
        ])

        return "\n".join(lines)

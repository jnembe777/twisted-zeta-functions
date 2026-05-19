"""
Defect computation module with arbitrary precision.

The defect δ_φ(a,b) = φ(a)φ(b) - φ(ab) quantifies the failure of an
arithmetic transfer to preserve multiplication. This module provides
high-precision computation and verification of the twisted cocycle identity.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
from mpmath import mp, mpf, fabs
from arithmetics.core.transfer import ArithmeticTransfer


def compute_defect(
    phi: ArithmeticTransfer,
    a: Union[int, mpf],
    b: Union[int, mpf],
    precision: int = 50
) -> mpf:
    """
    Compute the defect δ_φ(a,b) = φ(a)φ(b) - φ(ab).

    Args:
        phi: The arithmetic transfer function
        a, b: Integers at which to evaluate the defect
        precision: Number of decimal places for mpmath (default: 50)

    Returns:
        The defect value as an arbitrary-precision mpf
    """
    mp.dps = precision
    phi.precision = precision

    phi_a = phi.phi(mpf(a))
    phi_b = phi.phi(mpf(b))
    phi_ab = phi.phi(mpf(a) * mpf(b))

    return phi_a * phi_b - phi_ab


def verify_cocycle_identity(
    phi: ArithmeticTransfer,
    a: Union[int, mpf],
    b: Union[int, mpf],
    c: Union[int, mpf],
    precision: int = 50,
    tolerance: Optional[mpf] = None
) -> Tuple[bool, mpf]:
    """
    Verify the twisted cocycle identity (Theorem 2.2).

    The identity states:
    φ(a)δ(b,c) + δ(a,bc) = δ(a,b)φ(c) + δ(ab,c)

    Args:
        phi: The arithmetic transfer function
        a, b, c: Test values
        precision: Decimal places for computation
        tolerance: Maximum allowed deviation (default: 10^{-precision+5})

    Returns:
        Tuple of (is_valid, residual) where residual is LHS - RHS
    """
    mp.dps = precision
    phi.precision = precision

    if tolerance is None:
        tolerance = mpf(10) ** (-(precision - 5))

    # Compute φ values
    phi_a = phi.phi(a)
    phi_c = phi.phi(c)

    # Compute defects
    delta_bc = compute_defect(phi, b, c, precision)
    delta_a_bc = compute_defect(phi, a, b * c, precision)
    delta_ab = compute_defect(phi, a, b, precision)
    delta_ab_c = compute_defect(phi, a * b, c, precision)

    # LHS: φ(a)δ(b,c) + δ(a,bc)
    lhs = phi_a * delta_bc + delta_a_bc

    # RHS: δ(a,b)φ(c) + δ(ab,c)
    rhs = delta_ab * phi_c + delta_ab_c

    residual = fabs(lhs - rhs)
    is_valid = residual < tolerance

    return is_valid, residual


class DefectMatrix:
    """
    Store and analyze defect values over a grid of test pairs.

    The defect matrix D[i,j] = δ_φ(a_i, b_j) provides a comprehensive
    view of the defect structure for a given transfer.
    """

    def __init__(
        self,
        phi: ArithmeticTransfer,
        a_range: Tuple[int, int] = (2, 10),
        b_range: Optional[Tuple[int, int]] = None,
        precision: int = 50
    ):
        """
        Initialize and compute the defect matrix.

        Args:
            phi: The arithmetic transfer function
            a_range: Range of a values as (min, max) inclusive
            b_range: Range of b values (defaults to a_range)
            precision: Decimal places for computation
        """
        self.phi = phi
        self.precision = precision
        self.a_range = a_range
        self.b_range = b_range if b_range else a_range

        mp.dps = precision
        phi.precision = precision

        # Generate test values
        self.a_values = list(range(a_range[0], a_range[1] + 1))
        self.b_values = list(range(self.b_range[0], self.b_range[1] + 1))

        # Compute defect matrix
        self._compute_matrix()

    def _compute_matrix(self):
        """Compute the full defect matrix."""
        n_a = len(self.a_values)
        n_b = len(self.b_values)

        # Store as list of lists of mpf for arbitrary precision
        self.matrix = []
        for i, a in enumerate(self.a_values):
            row = []
            for j, b in enumerate(self.b_values):
                delta = compute_defect(self.phi, a, b, self.precision)
                row.append(delta)
            self.matrix.append(row)

    def get(self, a: int, b: int) -> mpf:
        """
        Get the defect value for a specific pair.

        Args:
            a, b: Values to look up

        Returns:
            δ_φ(a, b)
        """
        if a not in self.a_values or b not in self.b_values:
            # Compute on demand if not in matrix
            return compute_defect(self.phi, a, b, self.precision)

        i = self.a_values.index(a)
        j = self.b_values.index(b)
        return self.matrix[i][j]

    def max_defect(self) -> Tuple[mpf, int, int]:
        """
        Find the maximum absolute defect value.

        Returns:
            Tuple of (max_value, a, b) where max occurs
        """
        max_val = mpf(0)
        max_a, max_b = self.a_values[0], self.b_values[0]

        for i, a in enumerate(self.a_values):
            for j, b in enumerate(self.b_values):
                val = fabs(self.matrix[i][j])
                if val > max_val:
                    max_val = val
                    max_a, max_b = a, b

        return max_val, max_a, max_b

    def min_defect(self) -> Tuple[mpf, int, int]:
        """
        Find the minimum absolute defect value (ignoring zeros).

        Returns:
            Tuple of (min_value, a, b) where min occurs
        """
        min_val = None
        min_a, min_b = self.a_values[0], self.b_values[0]

        for i, a in enumerate(self.a_values):
            for j, b in enumerate(self.b_values):
                val = fabs(self.matrix[i][j])
                if val > 0 and (min_val is None or val < min_val):
                    min_val = val
                    min_a, min_b = a, b

        return min_val if min_val else mpf(0), min_a, min_b

    def mean_defect(self) -> mpf:
        """Compute the mean absolute defect."""
        total = mpf(0)
        count = 0

        for row in self.matrix:
            for val in row:
                total += fabs(val)
                count += 1

        return total / count if count > 0 else mpf(0)

    def is_zero_defect(self, tolerance: Optional[mpf] = None) -> bool:
        """
        Check if all defects are effectively zero.

        Args:
            tolerance: Maximum value to consider as zero

        Returns:
            True if all defects are below tolerance
        """
        if tolerance is None:
            tolerance = mpf(10) ** (-(self.precision - 5))

        for row in self.matrix:
            for val in row:
                if fabs(val) > tolerance:
                    return False
        return True

    def verify_all_cocycles(
        self,
        c_values: Optional[List[int]] = None,
        tolerance: Optional[mpf] = None
    ) -> Tuple[bool, List[Tuple[int, int, int, mpf]]]:
        """
        Verify cocycle identity for all (a,b,c) combinations.

        Args:
            c_values: Values for c (default: same as a_values)
            tolerance: Maximum allowed residual

        Returns:
            Tuple of (all_valid, list of failures as (a, b, c, residual))
        """
        if c_values is None:
            c_values = self.a_values

        failures = []

        for a in self.a_values:
            for b in self.b_values:
                for c in c_values:
                    valid, residual = verify_cocycle_identity(
                        self.phi, a, b, c, self.precision, tolerance
                    )
                    if not valid:
                        failures.append((a, b, c, residual))

        return len(failures) == 0, failures

    def to_numpy(self) -> np.ndarray:
        """
        Convert matrix to numpy array (with float precision loss).

        Returns:
            numpy array of defect values
        """
        return np.array([[float(val) for val in row] for row in self.matrix])

    def summary(self) -> dict:
        """
        Generate a summary of the defect matrix.

        Returns:
            Dictionary with statistics
        """
        max_val, max_a, max_b = self.max_defect()
        min_val, min_a, min_b = self.min_defect()

        return {
            'transfer_name': self.phi.name,
            'a_range': self.a_range,
            'b_range': self.b_range,
            'precision': self.precision,
            'max_defect': float(max_val),
            'max_at': (max_a, max_b),
            'min_defect': float(min_val),
            'min_at': (min_a, min_b),
            'mean_defect': float(self.mean_defect()),
            'is_zero_defect': self.is_zero_defect(),
            'n_pairs': len(self.a_values) * len(self.b_values),
        }

    def __repr__(self) -> str:
        return f"DefectMatrix(phi={self.phi.name}, size={len(self.a_values)}x{len(self.b_values)})"


def batch_compute_defects(
    phi: ArithmeticTransfer,
    pairs: List[Tuple[int, int]],
    precision: int = 50
) -> List[mpf]:
    """
    Compute defects for a batch of (a, b) pairs.

    Args:
        phi: The arithmetic transfer function
        pairs: List of (a, b) tuples
        precision: Decimal places for computation

    Returns:
        List of defect values in same order as pairs
    """
    return [compute_defect(phi, a, b, precision) for a, b in pairs]


def defect_gradient(
    phi: ArithmeticTransfer,
    a: int,
    b: int,
    precision: int = 50
) -> Tuple[mpf, mpf]:
    """
    Compute the discrete gradient of the defect at (a, b).

    Returns (∂δ/∂a, ∂δ/∂b) approximated by finite differences.

    Args:
        phi: The arithmetic transfer function
        a, b: Point at which to compute gradient
        precision: Decimal places for computation

    Returns:
        Tuple of (d_delta_da, d_delta_db)
    """
    mp.dps = precision

    delta_center = compute_defect(phi, a, b, precision)
    delta_a_plus = compute_defect(phi, a + 1, b, precision)
    delta_b_plus = compute_defect(phi, a, b + 1, precision)

    d_da = delta_a_plus - delta_center
    d_db = delta_b_plus - delta_center

    return d_da, d_db

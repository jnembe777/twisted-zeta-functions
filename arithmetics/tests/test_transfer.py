"""
Unit tests for arithmetic transfer classes.
"""

import math
import unittest
from mpmath import mp, mpf, fabs

from arithmetics.core.transfer import (
    ExponentialTransfer,
    IteratedExponentialTransfer,
    CoherentTwistTransfer,
    AffineTransfer,
    PolynomialTransfer,
    IdentityTransfer,
    MixedExponentialTransfer,
    RationalTransfer,
)


class TestExponentialTransfer(unittest.TestCase):
    """Tests for ExponentialTransfer."""

    def setUp(self):
        self.phi = ExponentialTransfer(alpha=2.0)
        self.phi.precision = 30
        mp.dps = 30

    def test_phi_computation(self):
        """Test that φ_α(n) = α^n."""
        for n in [1, 2, 3, 5, 10]:
            expected = mpf(2) ** n
            actual = self.phi.phi(n)
            self.assertAlmostEqual(float(actual), float(expected), places=20)

    def test_phi_inverse(self):
        """Test that φ^{-1}(α^n) = n."""
        for n in [1, 2, 3, 5, 10]:
            y = self.phi.phi(n)
            n_recovered = self.phi.phi_inverse(y)
            self.assertAlmostEqual(float(n_recovered), float(n), places=15)

    def test_invalid_alpha(self):
        """Test that α ≤ 0 raises error."""
        with self.assertRaises(ValueError):
            ExponentialTransfer(alpha=0)
        with self.assertRaises(ValueError):
            ExponentialTransfer(alpha=-1)
        with self.assertRaises(ValueError):
            ExponentialTransfer(alpha=1)  # α = 1 is trivial

    def test_twisted_multiply(self):
        """Test twisted multiplication a ⊗_φ b."""
        # For exponential: a ⊗_φ b = log_α(α^a · α^b) = a + b
        result = self.phi.twisted_multiply(3, 4)
        # φ^{-1}(φ(3) · φ(4)) = log_2(2^3 · 2^4) = log_2(2^7) = 7
        self.assertAlmostEqual(float(result), 7.0, places=10)


class TestIteratedExponentialTransfer(unittest.TestCase):
    """Tests for IteratedExponentialTransfer."""

    def test_level_0_is_identity(self):
        """Test that level 0 is the identity."""
        phi = IteratedExponentialTransfer(base=math.e, level=0)
        phi.precision = 30
        for n in [1, 2, 5, 10]:
            self.assertAlmostEqual(float(phi.phi(n)), float(n), places=15)

    def test_level_1_is_exp(self):
        """Test that level 1 is exp."""
        phi = IteratedExponentialTransfer(base=math.e, level=1)
        phi.precision = 30
        for n in [1, 2, 3]:
            expected = math.e ** n
            actual = float(phi.phi(n))
            self.assertAlmostEqual(actual, expected, places=10)

    def test_oplus_operation(self):
        """Test x ⊕_n y operation."""
        phi = IteratedExponentialTransfer(base=math.e, level=1)
        phi.precision = 30
        # x ⊕_1 y = exp(log(x) + log(y)) = x · y
        result = phi.oplus(2, 3)
        self.assertAlmostEqual(float(result), 6.0, places=10)


class TestCoherentTwistTransfer(unittest.TestCase):
    """Tests for CoherentTwistTransfer."""

    def test_log_transfer(self):
        """Test log bijection transfer."""
        phi = CoherentTwistTransfer.log_transfer()
        phi.precision = 30
        # φ(e) = log(e) = 1
        result = phi.phi(math.e)
        self.assertAlmostEqual(float(result), 1.0, places=10)

    def test_square_transfer(self):
        """Test square bijection transfer."""
        phi = CoherentTwistTransfer.square_transfer()
        phi.precision = 30
        result = phi.phi(5)
        self.assertAlmostEqual(float(result), 25.0, places=10)

    def test_inverse_roundtrip(self):
        """Test φ^{-1}(φ(n)) = n."""
        phi = CoherentTwistTransfer.sqrt_transfer()
        phi.precision = 30
        for n in [4, 9, 16, 25]:
            y = phi.phi(n)
            n_recovered = phi.phi_inverse(y)
            self.assertAlmostEqual(float(n_recovered), float(n), places=10)


class TestAffineTransfer(unittest.TestCase):
    """Tests for AffineTransfer."""

    def test_scaling(self):
        """Test pure scaling φ(n) = cn."""
        phi = AffineTransfer(c=2.0, d=0.0)
        phi.precision = 30
        self.assertAlmostEqual(float(phi.phi(5)), 10.0, places=15)

    def test_translation(self):
        """Test affine with translation φ(n) = n + d."""
        phi = AffineTransfer(c=1.0, d=3.0)
        phi.precision = 30
        self.assertAlmostEqual(float(phi.phi(5)), 8.0, places=15)

    def test_inverse(self):
        """Test inverse computation."""
        phi = AffineTransfer(c=2.0, d=1.0)
        phi.precision = 30
        # φ(n) = 2n + 1, so φ^{-1}(y) = (y-1)/2
        result = phi.phi_inverse(7)  # (7-1)/2 = 3
        self.assertAlmostEqual(float(result), 3.0, places=15)

    def test_zero_coefficient_error(self):
        """Test that c=0 raises error."""
        with self.assertRaises(ValueError):
            AffineTransfer(c=0, d=1)


class TestPolynomialTransfer(unittest.TestCase):
    """Tests for PolynomialTransfer."""

    def test_quadratic(self):
        """Test φ(n) = n²."""
        phi = PolynomialTransfer([0, 0, 1])  # 0 + 0·n + 1·n²
        phi.precision = 30
        self.assertAlmostEqual(float(phi.phi(5)), 25.0, places=15)

    def test_cubic(self):
        """Test φ(n) = n³."""
        phi = PolynomialTransfer([0, 0, 0, 1])
        phi.precision = 30
        self.assertAlmostEqual(float(phi.phi(3)), 27.0, places=15)

    def test_mixed_polynomial(self):
        """Test φ(n) = n² + n + 1."""
        phi = PolynomialTransfer([1, 1, 1])
        phi.precision = 30
        # φ(3) = 9 + 3 + 1 = 13
        self.assertAlmostEqual(float(phi.phi(3)), 13.0, places=15)


class TestIdentityTransfer(unittest.TestCase):
    """Tests for IdentityTransfer."""

    def test_identity(self):
        """Test φ(n) = n."""
        phi = IdentityTransfer()
        phi.precision = 30
        for n in [1, 5, 10, 100]:
            self.assertAlmostEqual(float(phi.phi(n)), float(n), places=15)

    def test_inverse_identity(self):
        """Test φ^{-1}(n) = n."""
        phi = IdentityTransfer()
        phi.precision = 30
        for n in [1, 5, 10, 100]:
            self.assertAlmostEqual(float(phi.phi_inverse(n)), float(n), places=15)


class TestMixedExponentialTransfer(unittest.TestCase):
    """Tests for MixedExponentialTransfer."""

    def test_sum_form(self):
        """Test φ(n) = α^n + β·n."""
        phi = MixedExponentialTransfer(alpha=2.0, beta=1.0, form="alpha^n + beta*n")
        phi.precision = 30
        # φ(3) = 2³ + 1·3 = 8 + 3 = 11
        self.assertAlmostEqual(float(phi.phi(3)), 11.0, places=10)

    def test_product_form(self):
        """Test φ(n) = α^n · β^n = (αβ)^n."""
        phi = MixedExponentialTransfer(alpha=2.0, beta=3.0, form="alpha^n * beta^n")
        phi.precision = 30
        # φ(2) = 2² · 3² = 4 · 9 = 36
        self.assertAlmostEqual(float(phi.phi(2)), 36.0, places=10)


class TestRationalTransfer(unittest.TestCase):
    """Tests for RationalTransfer."""

    def test_rational(self):
        """Test φ(n) = n/(n+c)."""
        phi = RationalTransfer(c=1.0)
        phi.precision = 30
        # φ(3) = 3/(3+1) = 3/4 = 0.75
        self.assertAlmostEqual(float(phi.phi(3)), 0.75, places=15)

    def test_inverse(self):
        """Test inverse φ^{-1}(y) = cy/(1-y)."""
        phi = RationalTransfer(c=1.0)
        phi.precision = 30
        # φ^{-1}(0.75) = 1·0.75/(1-0.75) = 0.75/0.25 = 3
        result = phi.phi_inverse(0.75)
        self.assertAlmostEqual(float(result), 3.0, places=10)


if __name__ == '__main__':
    unittest.main()

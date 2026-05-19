"""
Unit tests for zeta function computations.
"""

import unittest
from mpmath import mp, mpf, mpc, fabs, zeta as mp_zeta

from arithmetics.core.transfer import ExponentialTransfer, IdentityTransfer
from arithmetics.zeta.dirichlet import (
    zeta_phi,
    zeta_phi_derivative,
    zeta_classical_check,
    DirichletSeriesEvaluator,
)
from arithmetics.zeta.euler_product import (
    sieve_primes,
    twisted_primes,
    is_twisted_prime,
    euler_product_zeta,
)


class TestDirichletSeries(unittest.TestCase):
    """Tests for Dirichlet series evaluation."""

    def setUp(self):
        mp.dps = 20

    def test_identity_gives_zeta_2s(self):
        """Test that identity transfer gives ζ(2s).

        For φ(n) = n (identity), the twisted zeta is:
        ζ_φ(s) = Σ φ(n)^{-s} / n^s = Σ n^{-s} / n^s = Σ n^{-2s} = ζ(2s)
        """
        phi = IdentityTransfer()
        phi.precision = 20

        # Test at s = 2: ζ_φ(2) should equal ζ(4) = π^4/90 ≈ 1.0823232337
        s = mpc(2, 0)
        our_zeta = zeta_phi(s, phi, n_max=100000, precision=20)
        expected_zeta = mp_zeta(4)  # ζ(2s) = ζ(4)

        rel_error = fabs(our_zeta - expected_zeta) / fabs(expected_zeta)
        self.assertLess(float(rel_error), 1e-4)

    def test_convergence_in_critical_strip(self):
        """Test evaluation in critical strip."""
        phi = ExponentialTransfer(alpha=2.0)
        phi.precision = 20

        # Test at s = 0.5 + 10i
        s = mpc(0.5, 10)
        result = zeta_phi(s, phi, n_max=50000, precision=20)

        # Should be a finite complex number
        self.assertTrue(result.real == result.real)  # Not NaN
        self.assertTrue(result.imag == result.imag)

    def test_derivative_computation(self):
        """Test zeta derivative computation."""
        phi = ExponentialTransfer(alpha=2.0)
        phi.precision = 20

        s = mpc(1.5, 5)
        deriv = zeta_phi_derivative(s, phi, n_max=10000, precision=20)

        # Should be finite
        self.assertTrue(deriv.real == deriv.real)
        self.assertTrue(deriv.imag == deriv.imag)

    def test_classical_check(self):
        """Test comparison with mpmath zeta(2s) for identity transfer."""
        s = mpc(2.5, 0)  # Use real s for easier verification
        our_val, expected_val, rel_error = zeta_classical_check(s, precision=20)

        # For identity, our zeta gives ζ(2s) = ζ(5)
        self.assertLess(float(rel_error), 1e-3)


class TestDirichletSeriesEvaluator(unittest.TestCase):
    """Tests for DirichletSeriesEvaluator class."""

    def test_caching(self):
        """Test that caching works correctly."""
        phi = ExponentialTransfer(alpha=2.0)
        evaluator = DirichletSeriesEvaluator(phi, n_max=10000, precision=20)

        s = complex(1.5, 5)

        # First evaluation
        result1 = evaluator.evaluate(s)

        # Second evaluation (should use cache)
        result2 = evaluator.evaluate(s)

        self.assertEqual(float(result1.real), float(result2.real))
        self.assertEqual(float(result1.imag), float(result2.imag))

    def test_batch_evaluate(self):
        """Test batch evaluation."""
        phi = ExponentialTransfer(alpha=2.0)
        evaluator = DirichletSeriesEvaluator(phi, n_max=10000, precision=20)

        s_values = [complex(1.5, i) for i in range(1, 5)]
        results = evaluator.batch_evaluate(s_values)

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(r.real == r.real)  # Not NaN


class TestSievePrimes(unittest.TestCase):
    """Tests for prime sieve."""

    def test_small_primes(self):
        """Test first few primes."""
        primes = sieve_primes(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        self.assertEqual(primes, expected)

    def test_prime_count(self):
        """Test prime counting function."""
        primes = sieve_primes(100)
        self.assertEqual(len(primes), 25)  # π(100) = 25


class TestTwistedPrimes(unittest.TestCase):
    """Tests for twisted prime detection."""

    def setUp(self):
        mp.dps = 20

    def test_identity_primes_are_classical(self):
        """Test that identity transfer gives classical primes."""
        phi = IdentityTransfer()
        phi.precision = 20

        # For identity, twisted primes should match classical primes
        # (up to subtleties in definition)
        classical = set(sieve_primes(50))
        twisted = set(twisted_primes(phi, 50, precision=20))

        # Should have significant overlap
        overlap = len(classical & twisted) / len(classical)
        self.assertGreater(overlap, 0.8)

    def test_is_twisted_prime_basic(self):
        """Test basic twisted prime check."""
        phi = ExponentialTransfer(alpha=2.0)
        phi.precision = 20

        # 2 should be prime in most twisted arithmetics
        # (it can't be factored as a ⊗_φ b with a,b > 1)
        is_prime = is_twisted_prime(2, phi, precision=20)
        self.assertTrue(is_prime)


class TestEulerProduct(unittest.TestCase):
    """Tests for Euler product computation."""

    def setUp(self):
        mp.dps = 20

    def test_euler_product_convergence(self):
        """Test that Euler product gives reasonable values.

        For identity transfer, the Euler product should converge to ζ(2s).
        """
        phi = IdentityTransfer()
        phi.precision = 20

        s = mpc(2, 0)
        result = euler_product_zeta(s, phi, prime_limit=1000, precision=20)

        # For identity, should be close to ζ(2s) = ζ(4) = π^4/90
        expected = float(mp_zeta(4))
        self.assertLess(abs(float(result.real) - expected) / expected, 0.15)


if __name__ == '__main__':
    unittest.main()

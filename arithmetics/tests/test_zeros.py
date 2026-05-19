"""
Unit tests for zero finding module.
"""

import unittest
from mpmath import mp, mpf, mpc, fabs

from arithmetics.core.transfer import ExponentialTransfer, IdentityTransfer
from arithmetics.zeta.zeros import (
    newton_raphson_complex,
    secant_method_complex,
    argument_principle_count,
    grid_search_zeros,
    find_zeros,
    ZeroSearchRegion,
    ZeroFinder,
)


class TestNewtonRaphsonComplex(unittest.TestCase):
    """Tests for Newton-Raphson in complex plane."""

    def setUp(self):
        mp.dps = 30

    def test_find_simple_zero(self):
        """Test finding zero of z² - 1."""
        def f(z):
            return z * z - 1

        def f_prime(z):
            return 2 * z

        # Starting near z = 1
        zero, n_iter, error = newton_raphson_complex(
            f, f_prime, mpc(0.9, 0.1), tol=1e-15, max_iter=50, precision=30
        )

        self.assertIsNotNone(zero)
        self.assertLess(float(fabs(zero - 1)), 1e-10)

    def test_find_complex_zero(self):
        """Test finding complex zero."""
        def f(z):
            return z * z + 1  # Zeros at ±i

        def f_prime(z):
            return 2 * z

        # Starting near z = i
        zero, n_iter, error = newton_raphson_complex(
            f, f_prime, mpc(0.1, 0.9), tol=1e-15, max_iter=50, precision=30
        )

        self.assertIsNotNone(zero)
        self.assertLess(float(fabs(zero - mpc(0, 1))), 1e-10)


class TestSecantMethod(unittest.TestCase):
    """Tests for secant method."""

    def setUp(self):
        mp.dps = 30

    def test_find_zero(self):
        """Test finding zero without derivative."""
        def f(z):
            return z * z - 4  # Zeros at ±2

        zero, n_iter, error = secant_method_complex(
            f, mpc(1.5, 0), mpc(2.5, 0), tol=1e-15, max_iter=50, precision=30
        )

        self.assertIsNotNone(zero)
        self.assertLess(float(fabs(zero - 2)), 1e-10)


class TestArgumentPrinciple(unittest.TestCase):
    """Tests for argument principle zero counting."""

    def setUp(self):
        mp.dps = 20

    def test_count_single_zero(self):
        """Test counting a single simple zero."""
        def f(z):
            return z - mpc(0.5, 0.5)  # Single zero at 0.5 + 0.5i

        count = argument_principle_count(
            f, center=complex(0.5, 0.5), radius=0.1, n_points=100, precision=20
        )

        self.assertEqual(count, 1)

    def test_count_no_zeros(self):
        """Test counting with no zeros inside."""
        def f(z):
            return z + 10  # Zero at -10, far from our region

        count = argument_principle_count(
            f, center=complex(0, 0), radius=1, n_points=100, precision=20
        )

        self.assertEqual(count, 0)


class TestZeroSearchRegion(unittest.TestCase):
    """Tests for ZeroSearchRegion class."""

    def test_grid_points(self):
        """Test grid point generation."""
        region = ZeroSearchRegion(
            re_min=0.0, re_max=1.0,
            im_min=0.0, im_max=1.0,
            re_step=0.5, im_step=0.5
        )

        points = region.grid_points()

        # Should have 3 * 3 = 9 points
        self.assertEqual(len(points), 9)


class TestGridSearchZeros(unittest.TestCase):
    """Tests for grid search zero finding."""

    def setUp(self):
        mp.dps = 20

    def test_find_candidates(self):
        """Test finding zero candidates via grid search."""
        def f(z):
            return (z - mpc(0.5, 0.5)) * (z - mpc(0.5, 1.5))

        region = ZeroSearchRegion(
            re_min=0.0, re_max=1.0,
            im_min=0.0, im_max=2.0,
            re_step=0.1, im_step=0.1
        )

        candidates = grid_search_zeros(f, region, threshold=0.5, precision=20)

        # Should find candidates near both zeros
        self.assertGreater(len(candidates), 0)


class TestZeroFinder(unittest.TestCase):
    """Tests for ZeroFinder class."""

    def setUp(self):
        mp.dps = 15

    def test_initialization(self):
        """Test ZeroFinder initialization."""
        phi = ExponentialTransfer(alpha=2.0)
        finder = ZeroFinder(phi, precision=15, n_max=10000)

        self.assertEqual(finder.precision, 15)
        self.assertEqual(finder.n_max, 10000)

    def test_find_in_small_region(self):
        """Test finding zeros in a small region."""
        phi = ExponentialTransfer(alpha=2.0)
        finder = ZeroFinder(phi, precision=15, n_max=10000)

        region = ZeroSearchRegion(
            re_min=0.4, re_max=0.6,
            im_min=10.0, im_max=20.0,
            re_step=0.05, im_step=1.0
        )

        # This is a quick test - may or may not find zeros
        zeros = finder.find_in_region(region, tol=1e-10)

        # Just verify it runs without error
        self.assertIsInstance(zeros, list)

    def test_statistics(self):
        """Test statistics computation."""
        phi = ExponentialTransfer(alpha=2.0)
        finder = ZeroFinder(phi, precision=15, n_max=10000)

        stats = finder.statistics()

        self.assertIn('n_zeros', stats)

    def test_clear(self):
        """Test clearing found zeros."""
        phi = ExponentialTransfer(alpha=2.0)
        finder = ZeroFinder(phi, precision=15, n_max=10000)

        finder.clear()

        self.assertEqual(len(finder.all_zeros()), 0)


class TestFindZerosIntegration(unittest.TestCase):
    """Integration tests for zero finding."""

    def setUp(self):
        mp.dps = 15

    def test_find_zeros_basic(self):
        """Basic integration test for find_zeros function."""
        phi = ExponentialTransfer(alpha=2.0)

        region = ZeroSearchRegion(
            re_min=0.4, re_max=0.6,
            im_min=5.0, im_max=15.0,
            re_step=0.05, im_step=1.0
        )

        # Quick test with low precision and few terms
        zeros = find_zeros(
            phi, region,
            tol=1e-10,
            max_iter=50,
            precision=15,
            n_max=10000,
            verify=False
        )

        # Just verify it returns a list
        self.assertIsInstance(zeros, list)


if __name__ == '__main__':
    unittest.main()

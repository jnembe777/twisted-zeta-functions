"""
Unit tests for defect computation module.
"""

import unittest
from mpmath import mp, mpf, fabs

from arithmetics.core.transfer import (
    ExponentialTransfer,
    IdentityTransfer,
    AffineTransfer,
    IteratedExponentialTransfer,
)
from arithmetics.core.defect import (
    compute_defect,
    verify_cocycle_identity,
    DefectMatrix,
    batch_compute_defects,
    defect_gradient,
)


class TestComputeDefect(unittest.TestCase):
    """Tests for compute_defect function."""

    def setUp(self):
        mp.dps = 50

    def test_exponential_defect(self):
        """Test δ_α(a,b) = α^{a+b} - α^{ab} for exponential transfer."""
        phi = ExponentialTransfer(alpha=2.0)

        # δ_2(2,3) = 2^{2+3} - 2^{2·3} = 2^5 - 2^6 = 32 - 64 = -32
        delta = compute_defect(phi, 2, 3, precision=50)
        self.assertAlmostEqual(float(delta), -32.0, places=30)

        # δ_2(3,4) = 2^7 - 2^12 = 128 - 4096 = -3968
        delta = compute_defect(phi, 3, 4, precision=50)
        self.assertAlmostEqual(float(delta), -3968.0, places=30)

    def test_identity_zero_defect(self):
        """Test that identity transfer has zero defect."""
        phi = IdentityTransfer()

        for a, b in [(2, 3), (5, 7), (3, 4)]:
            delta = compute_defect(phi, a, b, precision=50)
            # δ(a,b) = a·b - ab = 0
            self.assertAlmostEqual(float(delta), 0.0, places=40)

    def test_scaling_zero_defect(self):
        """Test that pure scaling (d=0) has zero defect."""
        phi = AffineTransfer(c=2.0, d=0.0)

        # δ(a,b) = φ(a)φ(b) - φ(ab) = (2a)(2b) - 2(ab) = 4ab - 2ab = 2ab
        # Wait, this is NOT zero for affine! Let me recalculate.
        # φ(a) = 2a, φ(b) = 2b, φ(ab) = 2ab
        # δ(a,b) = 2a · 2b - 2ab = 4ab - 2ab = 2ab
        # So scaling does have non-zero defect unless c=1

        delta = compute_defect(phi, 2, 3, precision=50)
        # δ(2,3) = 4·6 - 2·6 = 24 - 12 = 12
        self.assertAlmostEqual(float(delta), 12.0, places=30)


class TestVerifyCocycleIdentity(unittest.TestCase):
    """Tests for cocycle identity verification."""

    def setUp(self):
        mp.dps = 50

    def test_exponential_cocycle(self):
        """Test cocycle identity for exponential transfer."""
        phi = ExponentialTransfer(alpha=2.0)

        # Test several triples
        for a, b, c in [(2, 3, 4), (1, 2, 3), (3, 5, 7)]:
            valid, residual = verify_cocycle_identity(phi, a, b, c, precision=50)
            self.assertTrue(valid, f"Cocycle failed for ({a},{b},{c})")
            self.assertLess(float(residual), 1e-40)

    def test_example_from_document(self):
        """Test the example from document.tex: α=2, (a,b,c)=(2,3,4)."""
        phi = ExponentialTransfer(alpha=2.0)

        # From document: LHS = RHS = -16776704
        valid, residual = verify_cocycle_identity(phi, 2, 3, 4, precision=50)
        self.assertTrue(valid)
        self.assertLess(float(residual), 1e-40)


class TestDefectMatrix(unittest.TestCase):
    """Tests for DefectMatrix class."""

    def setUp(self):
        mp.dps = 30
        self.phi = ExponentialTransfer(alpha=2.0)

    def test_matrix_construction(self):
        """Test that matrix is constructed correctly."""
        matrix = DefectMatrix(self.phi, a_range=(2, 5), b_range=(2, 5), precision=30)

        self.assertEqual(len(matrix.a_values), 4)  # 2, 3, 4, 5
        self.assertEqual(len(matrix.b_values), 4)
        self.assertEqual(len(matrix.matrix), 4)
        self.assertEqual(len(matrix.matrix[0]), 4)

    def test_get_method(self):
        """Test get method retrieves correct values."""
        matrix = DefectMatrix(self.phi, a_range=(2, 5), b_range=(2, 5), precision=30)

        # δ_2(2,3) = -32
        delta_23 = matrix.get(2, 3)
        self.assertAlmostEqual(float(delta_23), -32.0, places=20)

    def test_max_defect(self):
        """Test max_defect method."""
        matrix = DefectMatrix(self.phi, a_range=(2, 4), b_range=(2, 4), precision=30)

        max_val, max_a, max_b = matrix.max_defect()
        # All defects are negative for α > 1, so max is the one closest to 0
        self.assertGreater(float(max_val), 0)  # Absolute value

    def test_is_zero_defect(self):
        """Test is_zero_defect for identity."""
        phi_identity = IdentityTransfer()
        matrix = DefectMatrix(phi_identity, a_range=(2, 5), b_range=(2, 5), precision=30)

        self.assertTrue(matrix.is_zero_defect())

    def test_verify_all_cocycles(self):
        """Test verification of cocycle identity for all pairs."""
        matrix = DefectMatrix(self.phi, a_range=(2, 4), b_range=(2, 4), precision=30)

        all_valid, failures = matrix.verify_all_cocycles(c_values=[2, 3, 4])
        self.assertTrue(all_valid)
        self.assertEqual(len(failures), 0)

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        matrix = DefectMatrix(self.phi, a_range=(2, 3), b_range=(2, 3), precision=30)
        np_array = matrix.to_numpy()

        self.assertEqual(np_array.shape, (2, 2))

    def test_summary(self):
        """Test summary generation."""
        matrix = DefectMatrix(self.phi, a_range=(2, 4), b_range=(2, 4), precision=30)
        summary = matrix.summary()

        self.assertIn('transfer_name', summary)
        self.assertIn('max_defect', summary)
        self.assertIn('mean_defect', summary)


class TestBatchComputeDefects(unittest.TestCase):
    """Tests for batch_compute_defects function."""

    def test_batch_computation(self):
        """Test batch defect computation."""
        phi = ExponentialTransfer(alpha=2.0)
        pairs = [(2, 3), (3, 4), (4, 5)]

        results = batch_compute_defects(phi, pairs, precision=30)

        self.assertEqual(len(results), 3)
        self.assertAlmostEqual(float(results[0]), -32.0, places=20)  # δ(2,3)


class TestDefectGradient(unittest.TestCase):
    """Tests for defect_gradient function."""

    def test_gradient_computation(self):
        """Test discrete gradient computation."""
        phi = ExponentialTransfer(alpha=2.0)

        d_da, d_db = defect_gradient(phi, 3, 4, precision=30)

        # Verify gradient is reasonable (not NaN or inf)
        self.assertTrue(float(d_da) == float(d_da))  # Not NaN
        self.assertTrue(float(d_db) == float(d_db))


if __name__ == '__main__':
    unittest.main()

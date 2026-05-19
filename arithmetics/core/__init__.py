"""
Core module for arithmetic transfer operations.

Contains:
- transfer: ArithmeticTransfer base class and implementations
- defect: Defect δ_φ(a,b) computation with arbitrary precision
- cohomology: Projection onto H²_reg sector
"""

from arithmetics.core.transfer import (
    ArithmeticTransfer,
    ExponentialTransfer,
    IteratedExponentialTransfer,
    CoherentTwistTransfer,
    AffineTransfer,
    PolynomialTransfer,
)
from arithmetics.core.defect import compute_defect, DefectMatrix, verify_cocycle_identity
from arithmetics.core.cohomology import (
    project_to_regular_sector,
    compute_regular_norm,
    classify_defect,
)

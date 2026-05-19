"""
Zeta module for twisted zeta function computations.

Contains:
- dirichlet: ζ_φ(s) via Dirichlet series summation
- euler_product: Twisted Euler product (when applicable)
- zeros: Newton-Raphson zero finding in ℂ
"""

from arithmetics.zeta.dirichlet import zeta_phi, zeta_phi_derivative
from arithmetics.zeta.euler_product import twisted_primes, euler_product_zeta
from arithmetics.zeta.zeros import find_zeros, verify_zero, ZeroSearchRegion

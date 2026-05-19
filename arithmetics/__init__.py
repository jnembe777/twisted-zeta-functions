"""
Twisted Zeta Functions Experimental Framework

A computational framework for studying zeros of twisted zeta functions
across arithmetic deformations.
"""

__version__ = "0.1.0"
__author__ = "Jocelyn Nembe"

from arithmetics.core import transfer, defect, cohomology
from arithmetics.zeta import dirichlet, euler_product, zeros
from arithmetics.analytics import chebyshev, distribution, visualization

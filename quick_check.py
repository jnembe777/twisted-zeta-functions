"""
Quick check of zeta values at specific points.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, sqrt
from arithmetics.core.transfer import CoherentTwistTransfer, ExponentialTransfer
from arithmetics.zeta.dirichlet import zeta_phi

mp.dps = 20

print("QUICK ZETA CHECK")
print("=" * 60)

# Test sqrt arithmetic
phi = CoherentTwistTransfer(sqrt, lambda y: y*y, "sqrt")
phi.precision = 20

print("\nCoherentTwist_sqrt at s = 0.5 + ti:")
print("-" * 60)
for t in [14.0, 14.1, 14.13, 14.134, 14.1347, 14.5, 15.0]:
    s = mpc(0.5, t)
    z = zeta_phi(s, phi, n_max=1000, precision=20, adaptive=False)
    print(f"  t = {t:7.4f}: |zeta| = {float(fabs(z)):.6e}, zeta = {float(z.real):.4f} + {float(z.imag):.4f}i")

print("\nCoherentTwist_sqrt at different Re(s), Im = 14.1347:")
print("-" * 60)
for re in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    s = mpc(re, 14.1347)
    z = zeta_phi(s, phi, n_max=1000, precision=20, adaptive=False)
    print(f"  Re = {re:.1f}: |zeta| = {float(fabs(z)):.6e}")

# Compare with exponential
print("\n" + "=" * 60)
print("Exponential_alpha_2.0 at s = 0.5 + ti:")
print("-" * 60)
phi2 = ExponentialTransfer(2.0)
phi2.precision = 20

for t in [14.0, 14.1, 14.13, 14.134, 14.1347, 14.5, 15.0]:
    s = mpc(0.5, t)
    z = zeta_phi(s, phi2, n_max=1000, precision=20, adaptive=False)
    print(f"  t = {t:7.4f}: |zeta| = {float(fabs(z)):.6e}")

print("\nDone!")

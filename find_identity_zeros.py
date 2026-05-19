"""
Find genuine zeros for Identity arithmetic.
zeta_phi(s) = zeta(2s), so zeros at s = rho/2 where rho is Riemann zero.
This means Re(s) = 0.25, Im(s) = Im(rho)/2
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, nstr
from arithmetics.core.transfer import IdentityTransfer
from arithmetics.zeta.dirichlet import zeta_phi

mp.dps = 40

def refine_zero(phi, re_start, im_start, precision=35, max_iter=50):
    """2D Newton refinement."""
    phi.precision = precision + 5
    mp.dps = precision + 5

    s = mpc(re_start, im_start)
    n_max = 5000
    tol = mp.mpf(10) ** (-(precision - 5))

    for iteration in range(max_iter):
        z = zeta_phi(s, phi, n_max=n_max, precision=precision, adaptive=True)
        mag = fabs(z)

        if iteration % 5 == 0:
            print(f"    Iter {iteration+1:2d}: s = {nstr(s.real, 10)} + {nstr(s.imag, 10)}i, |zeta| = {nstr(mag, 6)}")

        if mag < tol:
            print(f"    CONVERGED!")
            return s, mag, True

        # Numerical gradient
        h = mp.mpf(10) ** (-precision // 3)
        z_re = zeta_phi(s + h, phi, n_max=n_max, precision=precision, adaptive=True)
        z_im = zeta_phi(s + h*1j, phi, n_max=n_max, precision=precision, adaptive=True)

        dz_dre = (z_re - z) / h
        dz_dim = (z_im - z) / (h * 1j)

        # Solve 2x2 system: dz/dre * dre + dz/dim * dim = -z
        # Using pseudo-inverse approach
        det = dz_dre * dz_dim.conjugate() - dz_dim * dz_dre.conjugate()
        if fabs(det) < mp.mpf(10) ** (-precision):
            # Use single direction
            if fabs(dz_dre) > fabs(dz_dim):
                step = z / dz_dre
            else:
                step = z / dz_dim * 1j
        else:
            step = z / dz_dre  # Simplified

        # Limit step
        if fabs(step) > 0.2:
            step = step * (0.2 / fabs(step))

        s = s - step

    z_final = zeta_phi(s, phi, n_max=n_max, precision=precision, adaptive=True)
    return s, fabs(z_final), False


def main():
    print("=" * 70)
    print("FINDING GENUINE ZEROS: Identity Arithmetic")
    print("=" * 70)
    print()
    print("For phi(n) = n: zeta_phi(s) = zeta(2s)")
    print("Riemann zeros: rho = 0.5 + i*t where t = 14.134725, 21.022040, ...")
    print("So zeta_phi zeros at: s = rho/2 = 0.25 + i*(t/2)")
    print()

    phi = IdentityTransfer()

    # Expected zero locations
    riemann_t = [14.134725141734694, 21.022039638771555, 25.010857580145688,
                 30.424876125859513, 32.935061587739189]

    print("Expected zeros of zeta_phi(s):")
    for t in riemann_t:
        print(f"  s = 0.25 + {t/2:.10f}i")
    print()

    # Check |zeta| at expected locations
    print("-" * 70)
    print("Checking at expected zero locations (Re(s) = 0.25):")
    print("-" * 70)
    phi.precision = 30

    for t in riemann_t:
        s = mpc(0.25, t/2)
        z = zeta_phi(s, phi, n_max=5000, precision=30, adaptive=True)
        mag = float(fabs(z))
        print(f"  s = 0.25 + {t/2:.6f}i: |zeta| = {mag:.6e}")

    # Compare with Re(s) = 0.5
    print()
    print("-" * 70)
    print("Comparison with Re(s) = 0.5 (wrong location):")
    print("-" * 70)

    for t in riemann_t[:3]:
        s_correct = mpc(0.25, t/2)
        s_wrong = mpc(0.5, t/2)
        z_correct = zeta_phi(s_correct, phi, n_max=5000, precision=30, adaptive=True)
        z_wrong = zeta_phi(s_wrong, phi, n_max=5000, precision=30, adaptive=True)
        print(f"  Im = {t/2:.4f}:")
        print(f"    Re=0.25: |zeta| = {float(fabs(z_correct)):.6e}")
        print(f"    Re=0.50: |zeta| = {float(fabs(z_wrong)):.6e}")

    # Refine the first zero
    print()
    print("=" * 70)
    print("REFINING FIRST ZERO")
    print("=" * 70)

    t1 = riemann_t[0]
    s0 = mpc(0.25, t1/2)
    print(f"\nStarting from s = 0.25 + {t1/2:.6f}i")
    print()

    s_refined, mag_refined, converged = refine_zero(phi, 0.25, t1/2, precision=35)

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Zero found: s = {nstr(s_refined.real, 20)} + {nstr(s_refined.imag, 20)}i")
    print(f"|zeta_phi(s)| = {nstr(mag_refined, 10)}")
    print()
    print(f"Expected (rho/2):")
    print(f"  Re(s) = 0.25")
    print(f"  Im(s) = {t1/2:.15f}")
    print()
    print(f"Deviation from expected:")
    print(f"  |Re(s) - 0.25| = {abs(float(s_refined.real) - 0.25):.2e}")
    print(f"  |Im(s) - {t1/2:.6f}| = {abs(float(s_refined.imag) - t1/2):.2e}")

    if converged or mag_refined < 1e-10:
        print()
        print("*** GENUINE ZERO CONFIRMED! ***")


if __name__ == "__main__":
    main()

"""
Quick refinement of zeros for various arithmetics.
Focus on candidates from the survey.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, nstr, sqrt, power, exp, ln
from arithmetics.core.transfer import CoherentTwistTransfer, ExponentialTransfer, PolynomialTransfer
from arithmetics.zeta.dirichlet import zeta_phi

def refine_1d(phi, im_start, precision=25, max_iter=30):
    """Quick 1D refinement on critical line starting from im_start."""
    mp.dps = precision + 5
    phi.precision = precision + 5

    im = mp.mpf(im_start)
    n_max = 3000
    tol = mp.mpf(10) ** (-(precision - 10))

    for iteration in range(max_iter):
        s = mpc(0.5, im)
        z = zeta_phi(s, phi, n_max=n_max, precision=precision, adaptive=True)
        mag = fabs(z)

        if iteration % 5 == 0 or mag < 0.01:
            print(f"    Iter {iteration+1:2d}: Im = {nstr(im, 10)}, |zeta| = {nstr(mag, 6)}")

        if mag < tol:
            return float(im), float(mag), True

        # Numerical derivative
        h = mp.mpf(10) ** (-precision // 3)
        z_plus = zeta_phi(mpc(0.5, im + h), phi, n_max=n_max, precision=precision, adaptive=True)
        z_minus = zeta_phi(mpc(0.5, im - h), phi, n_max=n_max, precision=precision, adaptive=True)
        dz = (z_plus - z_minus) / (2 * h)

        if fabs(dz) < mp.mpf(10) ** (-precision):
            break

        # Newton step
        step = z / dz
        if fabs(step) > 2.0:
            step = step * (2.0 / fabs(step))
        im = im - step.imag

    s_final = mpc(0.5, im)
    z_final = zeta_phi(s_final, phi, n_max=n_max, precision=precision, adaptive=True)
    return float(im), float(fabs(z_final)), False


def search_minimum(phi, im_center, width=1.0, steps=21, precision=20):
    """Search for local minimum of |zeta| near im_center."""
    mp.dps = precision + 5
    phi.precision = precision + 5

    best_im = im_center
    best_mag = float('inf')

    for i in range(steps):
        im = im_center - width/2 + width * i / (steps - 1)
        s = mpc(0.5, im)
        try:
            z = zeta_phi(s, phi, n_max=2000, precision=precision, adaptive=True)
            mag = float(fabs(z))
            if mag < best_mag:
                best_mag = mag
                best_im = im
        except:
            pass

    return best_im, best_mag


def main():
    print("=" * 70)
    print("QUICK ZERO REFINEMENT")
    print("=" * 70)
    print()

    # Riemann zero locations for reference
    riemann_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]

    # Test several interesting arithmetics
    arithmetics = [
        ("CoherentTwist_sqrt", CoherentTwistTransfer(sqrt, lambda y: y*y, "sqrt")),
        ("Exponential_alpha_1.5", ExponentialTransfer(1.5)),
        ("Polynomial_x2_plus_x_plus_1", PolynomialTransfer([1, 1, 1])),
    ]

    for name, phi in arithmetics:
        print(f"\n{'='*70}")
        print(f"Arithmetic: {name}")
        print(f"{'='*70}")

        # Search near each Riemann zero
        print("\nSearching near Riemann zero locations:")
        candidates = []

        for rz in riemann_zeros[:3]:  # First 3 Riemann zeros
            print(f"\n  Near Im = {rz:.6f}:")
            best_im, best_mag = search_minimum(phi, rz, width=2.0, steps=41, precision=20)
            print(f"    Best: Im = {best_im:.6f}, |zeta| = {best_mag:.6e}")
            candidates.append((best_im, best_mag))

        # Try to refine the best candidate
        best = min(candidates, key=lambda x: x[1])
        if best[1] < 0.5:
            print(f"\n  Refining best candidate at Im = {best[0]:.6f}:")
            im_refined, mag_refined, converged = refine_1d(phi, best[0], precision=25)
            status = "CONVERGED" if converged else "did not converge"
            print(f"\n  Result ({status}):")
            print(f"    Im(s) = {im_refined:.10f}")
            print(f"    |zeta| = {mag_refined:.6e}")

            # Compare to nearest Riemann zero
            nearest_rz = min(riemann_zeros, key=lambda r: abs(r - im_refined))
            diff = im_refined - nearest_rz
            print(f"    Nearest Riemann zero: {nearest_rz:.6f}")
            print(f"    Difference: {diff:+.6f}")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

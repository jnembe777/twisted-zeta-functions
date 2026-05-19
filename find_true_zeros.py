"""
2D search for true zeros of twisted zeta functions.
Search in the complex plane, not just on critical line.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, nstr, sqrt
from arithmetics.core.transfer import CoherentTwistTransfer, ExponentialTransfer
from arithmetics.zeta.dirichlet import zeta_phi

def grid_search_2d(phi, re_range, im_range, steps=21, precision=20):
    """2D grid search for zeros."""
    mp.dps = precision + 5
    phi.precision = precision + 5

    re_min, re_max = re_range
    im_min, im_max = im_range

    best_s = None
    best_mag = float('inf')
    candidates = []

    for i in range(steps):
        re = re_min + (re_max - re_min) * i / (steps - 1)
        for j in range(steps):
            im = im_min + (im_max - im_min) * j / (steps - 1)
            s = mpc(re, im)
            try:
                z = zeta_phi(s, phi, n_max=2000, precision=precision, adaptive=True)
                mag = float(fabs(z))

                if mag < 0.1:
                    candidates.append((re, im, mag))

                if mag < best_mag:
                    best_mag = mag
                    best_s = s
            except:
                pass

    return best_s, best_mag, candidates


def refine_2d(phi, s0, precision=30, max_iter=50):
    """2D Newton-Raphson refinement."""
    mp.dps = precision + 5
    phi.precision = precision + 5

    s = mpc(s0)
    n_max = 4000
    tol = mp.mpf(10) ** (-(precision - 10))

    for iteration in range(max_iter):
        z = zeta_phi(s, phi, n_max=n_max, precision=precision, adaptive=True)
        mag = fabs(z)

        if iteration % 10 == 0:
            print(f"    Iter {iteration+1:2d}: s = {nstr(s.real, 8)} + {nstr(s.imag, 8)}i, |zeta| = {nstr(mag, 6)}")

        if mag < tol:
            print(f"    CONVERGED at iteration {iteration+1}!")
            return s, mag, True

        # Numerical gradient
        h = mp.mpf(10) ** (-precision // 3)
        z_re = zeta_phi(s + h, phi, n_max=n_max, precision=precision, adaptive=True)
        z_im = zeta_phi(s + h*1j, phi, n_max=n_max, precision=precision, adaptive=True)

        dz_dre = (z_re - z) / h
        dz_dim = (z_im - z) / (h * 1j)

        # Newton step: solve for ds where z + dz/ds * ds = 0
        # We use: ds_re * dz/dre + ds_im * dz/dim = -z
        # Simplified: ds = -z / dz_ds using one direction

        if fabs(dz_dre) > fabs(dz_dim):
            step = z / dz_dre
        else:
            step = z / dz_dim

        # Limit step size
        if fabs(step) > 1.0:
            step = step / fabs(step)

        s_new = s - step

        # Keep in reasonable bounds
        if s_new.real < 0:
            s_new = mpc(0.1, s_new.imag)
        if s_new.real > 3:
            s_new = mpc(2.9, s_new.imag)
        if s_new.imag < 1:
            s_new = mpc(s_new.real, 1)

        s = s_new

    z_final = zeta_phi(s, phi, n_max=n_max, precision=precision, adaptive=True)
    return s, fabs(z_final), False


def main():
    print("=" * 70)
    print("2D ZERO SEARCH FOR TWISTED ZETA FUNCTIONS")
    print("=" * 70)
    print()

    # Test with sqrt arithmetic
    phi = CoherentTwistTransfer(sqrt, lambda y: y*y, "sqrt")

    print("Arithmetic: CoherentTwist_sqrt")
    print("phi(n) = sqrt(n)")
    print()

    # Search in multiple regions
    regions = [
        ((0.3, 0.7), (12, 16), "Near first Riemann zero"),
        ((0.3, 0.7), (19, 23), "Near second Riemann zero"),
        ((0.3, 0.7), (23, 27), "Near third Riemann zero"),
        ((0.7, 1.3), (12, 16), "Right of critical line, Im~14"),
        ((0.1, 0.5), (12, 16), "Left of critical line, Im~14"),
    ]

    all_candidates = []

    for re_range, im_range, description in regions:
        print(f"\n{'-'*70}")
        print(f"Region: Re in {re_range}, Im in {im_range}")
        print(f"Description: {description}")
        print(f"{'-'*70}")

        best_s, best_mag, candidates = grid_search_2d(
            phi, re_range, im_range, steps=31, precision=20
        )

        print(f"  Best point: s = {nstr(best_s.real, 6)} + {nstr(best_s.imag, 6)}i")
        print(f"  |zeta| = {best_mag:.6e}")

        if candidates:
            print(f"  Found {len(candidates)} points with |zeta| < 0.1")
            for re, im, mag in sorted(candidates, key=lambda x: x[2])[:3]:
                print(f"    s = {re:.4f} + {im:.4f}i, |zeta| = {mag:.6e}")
                all_candidates.append((re, im, mag))

    # Refine the best candidate
    if all_candidates:
        print()
        print("=" * 70)
        print("REFINING BEST CANDIDATE")
        print("=" * 70)

        best = min(all_candidates, key=lambda x: x[2])
        s0 = mpc(best[0], best[1])
        print(f"\nStarting from s = {best[0]:.6f} + {best[1]:.6f}i, |zeta| = {best[2]:.6e}")
        print()

        s_refined, mag_refined, converged = refine_2d(phi, s0, precision=30)

        print()
        print("=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Status: {'CONVERGED' if converged else 'Did not converge'}")
        print(f"Zero location: s = {nstr(s_refined.real, 12)} + {nstr(s_refined.imag, 12)}i")
        print(f"|zeta_phi(s)| = {nstr(mag_refined, 8)}")
        print()
        print(f"Distance from critical line: |Re(s) - 0.5| = {abs(float(s_refined.real) - 0.5):.6f}")

        # Check nearby Riemann zeros
        riemann_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
        nearest = min(riemann_zeros, key=lambda r: abs(r - float(s_refined.imag)))
        print(f"Nearest Riemann zero (Im): {nearest:.6f}")
        print(f"Difference in Im: {float(s_refined.imag) - nearest:+.6f}")


if __name__ == "__main__":
    main()

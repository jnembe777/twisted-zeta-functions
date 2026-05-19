"""
Systematic exploration of zeros for various arithmetics.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, nstr, sqrt, ln, exp
from arithmetics.core.transfer import (
    IdentityTransfer, AffineTransfer, ExponentialTransfer,
    CoherentTwistTransfer, PolynomialTransfer
)
from arithmetics.zeta.dirichlet import zeta_phi

mp.dps = 35

def grid_search_zeros(phi, re_range, im_range, grid_size=0.05, precision=25):
    """Find minima of |zeta| in a 2D grid."""
    phi.precision = precision
    re_min, re_max = re_range
    im_min, im_max = im_range

    best_points = []

    re = re_min
    while re <= re_max:
        im = im_min
        while im <= im_max:
            s = mpc(re, im)
            try:
                z = zeta_phi(s, phi, n_max=2000, precision=precision, adaptive=True)
                mag = float(fabs(z))
                if mag < 0.5:
                    best_points.append((re, im, mag))
            except:
                pass
            im += grid_size
        re += grid_size

    return sorted(best_points, key=lambda x: x[2])


def refine_2d(phi, re0, im0, precision=30, max_iter=40):
    """Refine zero using Newton-Raphson."""
    phi.precision = precision + 5
    mp.dps = precision + 5

    s = mpc(re0, im0)
    tol = mp.mpf(10) ** (-(precision - 5))

    for iteration in range(max_iter):
        z = zeta_phi(s, phi, n_max=4000, precision=precision, adaptive=True)
        mag = fabs(z)

        if mag < tol:
            return s, mag, True, iteration + 1

        h = mp.mpf(10) ** (-precision // 3)
        z_re = zeta_phi(s + h, phi, n_max=4000, precision=precision, adaptive=True)
        dz = (z_re - z) / h

        if fabs(dz) < mp.mpf(10) ** (-precision):
            break

        step = z / dz
        if fabs(step) > 0.3:
            step = step * (0.3 / fabs(step))

        s = s - step

        # Keep Re(s) reasonable
        if s.real < 0.1:
            s = mpc(0.1, s.imag)
        if s.real > 1.5:
            s = mpc(1.5, s.imag)

    z_final = zeta_phi(s, phi, n_max=4000, precision=precision, adaptive=True)
    return s, fabs(z_final), False, max_iter


def main():
    print("=" * 70)
    print("SYSTEMATIC ZERO EXPLORATION")
    print("=" * 70)

    results = []

    # Test multiple arithmetics
    arithmetics = [
        ("Identity (phi=n)", IdentityTransfer()),
        ("Affine_2n", AffineTransfer(2, 0)),
        ("Affine_n+1", AffineTransfer(1, 1)),
        ("Exponential_1.5", ExponentialTransfer(1.5)),
        ("Exponential_2.0", ExponentialTransfer(2.0)),
        ("CoherentTwist_sqrt", CoherentTwistTransfer(sqrt, lambda y: y*y, "sqrt")),
    ]

    for name, phi in arithmetics:
        print(f"\n{'='*70}")
        print(f"Arithmetic: {name}")
        print(f"{'='*70}")

        # Search in critical region
        print("\nGrid search in Re=[0.3, 0.7], Im=[5, 15]...")
        candidates = grid_search_zeros(phi, (0.3, 0.7), (5.0, 15.0), grid_size=0.1, precision=20)

        if candidates:
            print(f"Found {len(candidates)} candidate(s) with |zeta| < 0.5")
            for re, im, mag in candidates[:5]:
                print(f"  s = {re:.2f} + {im:.2f}i, |zeta| = {mag:.4f}")

            # Refine best candidate
            best = candidates[0]
            print(f"\nRefining best candidate...")
            s_ref, mag_ref, conv, iters = refine_2d(phi, best[0], best[1], precision=30)

            status = "ZERO!" if conv else ("likely" if mag_ref < 1e-6 else "minimum")
            print(f"  Result: s = {nstr(s_ref.real, 10)} + {nstr(s_ref.imag, 10)}i")
            print(f"  |zeta| = {nstr(mag_ref, 6)} [{status}]")

            if conv or mag_ref < 1e-6:
                results.append((name, s_ref, mag_ref))
        else:
            print("  No candidates found with |zeta| < 0.5")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: GENUINE ZEROS FOUND")
    print("=" * 70)

    if results:
        for name, s, mag in results:
            print(f"\n{name}:")
            print(f"  s = {nstr(s.real, 15)} + {nstr(s.imag, 15)}i")
            print(f"  |zeta_phi(s)| = {nstr(mag, 6)}")
    else:
        print("\nNo genuine zeros confirmed in the search region.")
        print("Try expanding the search range or increasing precision.")


if __name__ == "__main__":
    main()

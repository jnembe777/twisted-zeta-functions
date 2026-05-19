"""
Find arithmetics with genuine zeros of their twisted zeta functions.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, sqrt, ln, exp, nstr
from arithmetics.core.transfer import (
    ExponentialTransfer, IteratedExponentialTransfer,
    CoherentTwistTransfer, AffineTransfer, PolynomialTransfer,
    IdentityTransfer
)
from arithmetics.zeta.dirichlet import zeta_phi

def check_for_zeros(phi, name, im_values, precision=20):
    """
    Check for sign changes in zeta (indicating zeros).
    Returns list of (im, magnitude) for potential zeros.
    """
    mp.dps = precision + 5
    phi.precision = precision + 5

    candidates = []
    prev_z = None
    prev_im = None

    for im in im_values:
        s = mpc(0.5, im)
        try:
            z = zeta_phi(s, phi, n_max=2000, precision=precision, adaptive=True)
            mag = float(fabs(z))

            # Check for sign change (zero crossing)
            if prev_z is not None:
                if (prev_z.real * z.real < 0) or (prev_z.imag * z.imag < 0):
                    # Bisect to find better estimate
                    im_mid = (prev_im + im) / 2
                    s_mid = mpc(0.5, im_mid)
                    z_mid = zeta_phi(s_mid, phi, n_max=2000, precision=precision, adaptive=True)
                    mag_mid = float(fabs(z_mid))
                    candidates.append((im_mid, mag_mid, "crossing"))
                elif mag < 0.05:
                    candidates.append((im, mag, "small"))

            prev_z = z
            prev_im = im
        except:
            pass

    return candidates


def refine_zero(phi, im_start, precision=30, max_iter=50):
    """Refine a zero candidate using bisection + Newton."""
    mp.dps = precision + 5
    phi.precision = precision + 5

    # First, bracket the zero with bisection
    im = mp.mpf(im_start)
    best_im = im
    best_mag = float('inf')

    # Local search
    for delta in [d * 0.01 for d in range(-50, 51)]:
        test_im = float(im_start) + delta
        s = mpc(0.5, test_im)
        try:
            z = zeta_phi(s, phi, n_max=3000, precision=precision, adaptive=True)
            mag = float(fabs(z))
            if mag < best_mag:
                best_mag = mag
                best_im = test_im
        except:
            pass

    # Newton refinement from best point
    im = mp.mpf(best_im)
    for iteration in range(max_iter):
        s = mpc(0.5, im)
        z = zeta_phi(s, phi, n_max=4000, precision=precision, adaptive=True)
        mag = fabs(z)

        if mag < mp.mpf(10) ** (-(precision - 5)):
            return float(im), float(mag), True

        # Numerical derivative
        h = mp.mpf(10) ** (-precision // 3)
        z_plus = zeta_phi(mpc(0.5, im + h), phi, n_max=4000, precision=precision, adaptive=True)
        z_minus = zeta_phi(mpc(0.5, im - h), phi, n_max=4000, precision=precision, adaptive=True)
        dz = (z_plus - z_minus) / (2 * h)

        if fabs(dz) < mp.mpf(10) ** (-precision):
            break

        # Newton step (imaginary direction only)
        step = z / dz
        if fabs(step) > 0.5:
            step = step * (0.5 / fabs(step))
        im = im - step.imag

    s_final = mpc(0.5, im)
    z_final = zeta_phi(s_final, phi, n_max=4000, precision=precision, adaptive=True)
    return float(im), float(fabs(z_final)), False


def main():
    print("=" * 70)
    print("SEARCHING FOR ARITHMETICS WITH GENUINE ZEROS")
    print("=" * 70)
    print()

    # Fine grid of imaginary values
    im_values = [10.0 + i * 0.1 for i in range(400)]  # 10 to 50

    # Arithmetics to test (focusing on well-behaved ones)
    arithmetics = [
        ("Identity (phi=n)", IdentityTransfer()),
        ("Affine_2n", AffineTransfer(2, 0)),
        ("Affine_0.5n", AffineTransfer(0.5, 0)),
        ("Affine_n+1", AffineTransfer(1, 1)),
        ("Exponential_1.1", ExponentialTransfer(1.1)),
        ("Exponential_1.2", ExponentialTransfer(1.2)),
        ("Exponential_1.5", ExponentialTransfer(1.5)),
        ("Hierarchy_base2_level1", IteratedExponentialTransfer(2, 1)),
        ("Hierarchy_basee_level1", IteratedExponentialTransfer(mp.e, 1)),
        ("Polynomial_x+1", PolynomialTransfer([1, 1])),
        ("CoherentTwist_log", CoherentTwistTransfer(ln, exp, "log")),
        ("CoherentTwist_double", CoherentTwistTransfer(lambda x: 2*x, lambda y: y/2, "double")),
    ]

    results = []

    for name, phi in arithmetics:
        print(f"\n{'-'*70}")
        print(f"Testing: {name}")
        print(f"{'-'*70}")

        # Quick check of first few terms to ensure series is not degenerate
        mp.dps = 25
        phi.precision = 25
        term_mags = []
        for n in [1, 2, 3, 4, 5]:
            phi_n = phi.phi(n)
            s = mpc(0.5, 14.0)
            from mpmath import power
            term = power(phi_n, -s) / power(n, s)
            term_mags.append(float(fabs(term)))

        ratio = term_mags[1] / term_mags[0] if term_mags[0] > 0 else 0
        print(f"  Term ratio |a_2|/|a_1| = {ratio:.6f}")

        if ratio < 1e-10:
            print(f"  SKIPPING: Series degenerate (first term dominates)")
            continue

        # Search for zeros
        candidates = check_for_zeros(phi, name, im_values, precision=20)

        if candidates:
            print(f"  Found {len(candidates)} candidate(s):")
            for im, mag, ctype in candidates[:5]:
                print(f"    Im = {im:.4f}, |zeta| = {mag:.6e} ({ctype})")

            # Refine best candidate
            best = min(candidates, key=lambda x: x[1])
            print(f"\n  Refining best candidate at Im = {best[0]:.4f}...")

            im_refined, mag_refined, converged = refine_zero(phi, best[0], precision=30)

            status = "GENUINE ZERO!" if converged else ("likely zero" if mag_refined < 1e-6 else "local minimum")
            print(f"  Result: Im = {im_refined:.10f}, |zeta| = {mag_refined:.2e} [{status}]")

            results.append((name, im_refined, mag_refined, converged))
        else:
            print(f"  No candidates found in range [10, 50]")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: ARITHMETICS WITH GENUINE ZEROS")
    print("=" * 70)

    genuine = [(n, im, mag) for n, im, mag, conv in results if conv or mag < 1e-8]
    likely = [(n, im, mag) for n, im, mag, conv in results if not conv and mag < 1e-3 and mag >= 1e-8]

    if genuine:
        print("\nGENUINE ZEROS (|zeta| < 10^-8):")
        print("-" * 70)
        for name, im, mag in genuine:
            print(f"  {name:<35} Im = {im:.10f}, |zeta| = {mag:.2e}")

    if likely:
        print("\nLIKELY ZEROS (|zeta| < 10^-3):")
        print("-" * 70)
        for name, im, mag in likely:
            print(f"  {name:<35} Im = {im:.10f}, |zeta| = {mag:.2e}")

    if not genuine and not likely:
        print("\nNo genuine zeros found. Try searching with higher precision or different ranges.")

    return results


if __name__ == "__main__":
    main()

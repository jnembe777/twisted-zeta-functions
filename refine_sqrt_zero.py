"""
Refine zeros for CoherentTwist_sqrt - a more interesting arithmetic.
phi(n) = sqrt(n), which gives non-trivial series behavior.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, nstr, sqrt, power
from arithmetics.core.transfer import CoherentTwistTransfer
from arithmetics.zeta.dirichlet import zeta_phi

def scan_and_refine(phi, im_start=10.0, im_end=40.0, precision=30):
    """
    Scan for zeros and refine promising candidates.
    """
    mp.dps = precision + 5
    phi.precision = precision + 5

    print(f"Arithmetic: {phi.name}")
    print(f"Scanning Re(s) = 0.5, Im(s) in [{im_start}, {im_end}]")
    print(f"Precision: {precision} digits")
    print()

    # Phase 1: Coarse scan
    print("-" * 70)
    print("PHASE 1: Coarse Scan (step = 0.5)")
    print("-" * 70)

    candidates = []
    prev_z = None
    prev_s = None

    im = im_start
    step = 0.5
    while im <= im_end:
        s = mpc(0.5, im)
        try:
            z = zeta_phi(s, phi, n_max=5000, precision=precision, adaptive=True)
            mag = float(fabs(z))

            # Look for sign changes in real/imag parts (zero crossing)
            if prev_z is not None:
                # Check if we crossed zero (sign change)
                if (prev_z.real * z.real < 0) or (prev_z.imag * z.imag < 0):
                    candidates.append((float(prev_s.imag), float(im), float(fabs(prev_z)), mag))
                    print(f"  Zero crossing between Im = {prev_s.imag:.2f} and {im:.2f}")

            if mag < 0.5:
                print(f"  Im = {im:.2f}: |zeta| = {mag:.6e}")

            prev_z = z
            prev_s = s
        except Exception as e:
            print(f"  Im = {im:.2f}: ERROR - {e}")

        im += step

    print()

    # Phase 2: Fine search around each candidate
    print("-" * 70)
    print("PHASE 2: Fine Search Around Candidates")
    print("-" * 70)

    refined = []
    for im1, im2, mag1, mag2 in candidates:
        print(f"\nSearching between Im = {im1:.2f} and {im2:.2f}")

        best_im = im1
        best_mag = mag1

        # Binary-style search
        for refinement in range(3):
            step = (im2 - im1) / 20
            search_im = im1
            while search_im <= im2:
                s = mpc(0.5, search_im)
                try:
                    z = zeta_phi(s, phi, n_max=5000, precision=precision, adaptive=True)
                    mag = float(fabs(z))
                    if mag < best_mag:
                        best_mag = mag
                        best_im = search_im
                except:
                    pass
                search_im += step

            # Narrow the window
            im1 = best_im - step * 5
            im2 = best_im + step * 5

        print(f"  Best: Im = {best_im:.8f}, |zeta| = {best_mag:.6e}")
        refined.append((best_im, best_mag))

    # Phase 3: Newton refinement for best candidates
    print()
    print("-" * 70)
    print("PHASE 3: Newton-Raphson Refinement")
    print("-" * 70)

    final_zeros = []
    for im0, mag0 in sorted(refined, key=lambda x: x[1])[:5]:
        if mag0 < 0.3:  # Only refine promising candidates
            print(f"\nRefining candidate at Im = {im0:.6f} (initial |zeta| = {mag0:.6e})")

            im = mp.mpf(im0)
            for iteration in range(30):
                s = mpc(0.5, im)
                z = zeta_phi(s, phi, n_max=8000, precision=precision+10, adaptive=True)
                mag = fabs(z)

                # Numerical derivative
                h = mp.mpf(10) ** (-precision // 3)
                z_plus = zeta_phi(mpc(0.5, im + h), phi, n_max=8000, precision=precision+10, adaptive=True)
                z_minus = zeta_phi(mpc(0.5, im - h), phi, n_max=8000, precision=precision+10, adaptive=True)
                dz_di = (z_plus - z_minus) / (2 * h)

                if iteration % 5 == 0:
                    print(f"    Iter {iteration+1}: Im = {nstr(im, 12)}, |zeta| = {nstr(mag, 6)}")

                if mag < mp.mpf(10) ** (-(precision - 10)):
                    print(f"    CONVERGED at iteration {iteration+1}!")
                    final_zeros.append((float(im), float(mag)))
                    break

                if fabs(dz_di) > mp.mpf(10) ** (-precision):
                    # Newton step (1D along imaginary axis)
                    step = z.imag / dz_di.imag if abs(dz_di.imag) > abs(dz_di.real) else z.real / dz_di.real
                    if abs(step) > 1.0:
                        step = step / abs(step)
                    im = im - step
                else:
                    break

            else:
                final_zeros.append((float(im), float(fabs(zeta_phi(mpc(0.5, im), phi, n_max=8000, precision=precision+10, adaptive=True)))))

    return final_zeros


def main():
    print("=" * 70)
    print("ZERO FINDING: CoherentTwist_sqrt")
    print("=" * 70)
    print()

    # Create sqrt transfer: g(x) = sqrt(x), g^(-1)(y) = y^2
    phi = CoherentTwistTransfer(
        g=lambda x: sqrt(x),
        g_inverse=lambda y: y * y,
        name="sqrt"
    )
    phi.precision = 40

    # Show some phi values
    print("phi(n) = sqrt(n) values:")
    for n in [1, 2, 3, 4, 5, 10, 100]:
        print(f"  phi({n}) = {float(phi.phi(n)):.6f}")
    print()

    # Find zeros
    zeros = scan_and_refine(phi, im_start=10.0, im_end=50.0, precision=30)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if zeros:
        print("\nZeros found on critical line Re(s) = 0.5:")
        print("-" * 50)
        for im, mag in sorted(zeros, key=lambda x: x[0]):
            status = "CONFIRMED" if mag < 1e-10 else "CANDIDATE" if mag < 1e-5 else "WEAK"
            print(f"  Im(s) = {im:.10f}, |zeta| = {mag:.6e}  [{status}]")

        # Compare with Riemann zeros
        riemann_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918720, 43.327073, 48.005151]
        print("\nComparison with Riemann zeros:")
        print("-" * 50)
        for im, mag in sorted(zeros, key=lambda x: x[0]):
            closest_riemann = min(riemann_zeros, key=lambda r: abs(r - im))
            diff = im - closest_riemann
            print(f"  Twisted: {im:.6f}  vs  Riemann: {closest_riemann:.6f}  (diff = {diff:+.6f})")
    else:
        print("\nNo zeros found in the search region.")

    return zeros


if __name__ == "__main__":
    main()

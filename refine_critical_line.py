"""
Search for zeros specifically on or near the critical line Re(s) = 0.5
for Hierarchy_base10_level2.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, nstr
from arithmetics.core.transfer import IteratedExponentialTransfer
from arithmetics.zeta.dirichlet import zeta_phi

def scan_critical_line(phi, im_min=10.0, im_max=40.0, im_step=0.1, precision=20):
    """
    Scan along Re(s) = 0.5 looking for local minima of |zeta_phi(s)|.
    """
    mp.dps = precision + 5
    phi.precision = precision + 5

    print(f"Scanning critical line Re(s) = 0.5")
    print(f"  Im(s) range: [{im_min}, {im_max}]")
    print(f"  Step: {im_step}")
    print()

    results = []
    prev_mag = float('inf')
    prev_prev_mag = float('inf')

    im = im_min
    while im <= im_max:
        s = mpc(0.5, im)
        try:
            z = zeta_phi(s, phi, n_max=3000, precision=precision, adaptive=True)
            mag = float(fabs(z))

            # Detect local minimum
            if prev_mag < prev_prev_mag and prev_mag < mag:
                results.append((im - im_step, prev_mag))
                print(f"  Local minimum at Im(s) = {im - im_step:.4f}: |zeta| = {prev_mag:.6e}")

            prev_prev_mag = prev_mag
            prev_mag = mag

        except Exception as e:
            print(f"  Error at Im(s) = {im}: {e}")

        im += im_step

    return results


def refine_on_critical_line(phi, im0, precision=40, max_iter=50):
    """
    Refine a zero candidate, constraining Re(s) = 0.5.
    Uses 1D Newton-Raphson along imaginary axis only.
    """
    mp.dps = precision + 10
    phi.precision = precision + 10

    tol = mp.mpf(10) ** (-(precision - 10))
    im = mp.mpf(im0)
    n_max = 5000

    print(f"\nRefining on critical line starting from Im(s) = {im0}")
    print(f"  Tolerance: {tol}")

    for iteration in range(max_iter):
        s = mpc(0.5, im)
        z = zeta_phi(s, phi, n_max=n_max, precision=precision+5, adaptive=True)
        mag = fabs(z)

        # Numerical derivative w.r.t. imaginary part
        h = mp.mpf(10) ** (-precision // 3)
        z_plus = zeta_phi(mpc(0.5, im + h), phi, n_max=n_max, precision=precision+5, adaptive=True)
        z_minus = zeta_phi(mpc(0.5, im - h), phi, n_max=n_max, precision=precision+5, adaptive=True)

        # d|z|^2/dt = 2*Re(conj(z) * dz/dt)
        dz_di = (z_plus - z_minus) / (2 * h)

        print(f"  Iter {iteration+1:3d}: Im(s) = {nstr(im, 15)}, |zeta| = {nstr(mag, 8)}")

        if mag < tol:
            print(f"\nConverged!")
            return mpc(0.5, im), mag, iteration + 1

        # Newton step for finding zero of z (not |z|)
        # We want Im(z) = 0 and Re(z) = 0
        # Use: new_im = im - Im(z) / Im(dz/di)
        if fabs(dz_di) > mp.mpf(10) ** (-precision):
            # Step to minimize |z|
            step = z / (dz_di * 1j)  # dz/d(im) = dz/di * i
            step_im = step.imag

            # Limit step size
            if abs(step_im) > 1.0:
                step_im = step_im / abs(step_im)

            im_new = im - step_im

            # Check if improving
            z_new = zeta_phi(mpc(0.5, im_new), phi, n_max=n_max, precision=precision+5, adaptive=True)
            if fabs(z_new) >= mag:
                im_new = im - step_im * 0.1  # Smaller step

            im = im_new
        else:
            print("  Derivative too small")
            break

    s_final = mpc(0.5, im)
    z_final = zeta_phi(s_final, phi, n_max=n_max, precision=precision+5, adaptive=True)
    return s_final, fabs(z_final), max_iter


def check_off_critical(phi, re_values, im_value, precision=20):
    """
    Check |zeta| for various Re(s) at fixed Im(s).
    """
    mp.dps = precision + 5
    phi.precision = precision + 5

    print(f"\nChecking |zeta| at Im(s) = {im_value} for various Re(s):")
    print("-" * 50)

    best_re = None
    best_mag = float('inf')

    for re in re_values:
        s = mpc(re, im_value)
        try:
            z = zeta_phi(s, phi, n_max=3000, precision=precision, adaptive=True)
            mag = float(fabs(z))
            print(f"  Re(s) = {re:6.3f}: |zeta| = {mag:.6e}")

            if mag < best_mag:
                best_mag = mag
                best_re = re
        except:
            print(f"  Re(s) = {re:6.3f}: ERROR")

    print(f"\n  Best: Re(s) = {best_re}, |zeta| = {best_mag:.6e}")
    return best_re, best_mag


def main():
    print("=" * 70)
    print("CRITICAL LINE ANALYSIS: Hierarchy_base10_level2")
    print("=" * 70)
    print()

    phi = IteratedExponentialTransfer(base=10, level=2)
    print(f"Arithmetic: {phi.name}")
    print()

    # Phase 1: Coarse scan of critical line
    print("-" * 70)
    print("PHASE 1: Coarse Scan of Critical Line")
    print("-" * 70)
    minima = scan_critical_line(phi, im_min=10.0, im_max=40.0, im_step=0.5, precision=15)
    print()

    # Phase 2: Fine scan around each minimum
    print("-" * 70)
    print("PHASE 2: Fine Scan Around Minima")
    print("-" * 70)

    refined_minima = []
    for im0, mag0 in minima[:5]:  # Top 5 candidates
        print(f"\nFine scan around Im(s) = {im0:.2f}")
        fine_results = []
        for im in [im0 + d for d in [-0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4]]:
            s = mpc(0.5, im)
            try:
                mp.dps = 25
                phi.precision = 25
                z = zeta_phi(s, phi, n_max=3000, precision=25, adaptive=True)
                mag = float(fabs(z))
                fine_results.append((im, mag))
                print(f"    Im(s) = {im:.3f}: |zeta| = {mag:.6e}")
            except:
                pass

        if fine_results:
            best = min(fine_results, key=lambda x: x[1])
            refined_minima.append(best)
            print(f"  -> Best: Im(s) = {best[0]:.3f}, |zeta| = {best[1]:.6e}")

    # Phase 3: Check behavior across critical strip
    print()
    print("-" * 70)
    print("PHASE 3: Cross-section of Critical Strip")
    print("-" * 70)

    # Check at first Riemann zero height
    re_values = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    check_off_critical(phi, re_values, 14.1347, precision=20)

    # Check at the minimum we found
    if refined_minima:
        best_im = min(refined_minima, key=lambda x: x[1])[0]
        check_off_critical(phi, re_values, best_im, precision=20)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if refined_minima:
        best = min(refined_minima, key=lambda x: x[1])
        print(f"\nSmallest |zeta| on critical line Re(s) = 0.5:")
        print(f"  Im(s) = {best[0]:.6f}")
        print(f"  |zeta| = {best[1]:.6e}")

        if best[1] < 1e-3:
            print(f"\n  This appears to be a genuine zero near the critical line!")
        else:
            print(f"\n  This is a local minimum but NOT close to zero.")
            print(f"  The twisted zeta may not have zeros on Re(s) = 0.5 in this range.")

    return refined_minima


if __name__ == "__main__":
    main()

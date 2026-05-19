"""
Fast search for genuine zeros - optimized version.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, sqrt, ln, exp, nstr, power
from arithmetics.core.transfer import (
    ExponentialTransfer, IteratedExponentialTransfer,
    CoherentTwistTransfer, AffineTransfer, PolynomialTransfer,
    IdentityTransfer
)
from arithmetics.zeta.dirichlet import zeta_phi

mp.dps = 25

def quick_scan(phi, im_min=10, im_max=35, step=0.25):
    """Quick scan for sign changes."""
    phi.precision = 20

    prev_z = None
    prev_im = None
    crossings = []

    im = im_min
    while im <= im_max:
        s = mpc(0.5, im)
        try:
            z = zeta_phi(s, phi, n_max=1000, precision=20, adaptive=False)

            if prev_z is not None:
                # Check sign change in real or imaginary part
                if prev_z.real * z.real < 0:
                    crossings.append((prev_im, im, "real"))
                if prev_z.imag * z.imag < 0:
                    crossings.append((prev_im, im, "imag"))

            prev_z = z
            prev_im = im
        except:
            pass
        im += step

    return crossings


def bisect_zero(phi, im1, im2, tol=1e-6, max_iter=30):
    """Bisection to find zero between im1 and im2."""
    phi.precision = 25

    s1 = mpc(0.5, im1)
    s2 = mpc(0.5, im2)
    z1 = zeta_phi(s1, phi, n_max=2000, precision=25, adaptive=False)
    z2 = zeta_phi(s2, phi, n_max=2000, precision=25, adaptive=False)

    # Use real part for bisection
    use_real = (z1.real * z2.real < 0)

    for _ in range(max_iter):
        im_mid = (im1 + im2) / 2
        s_mid = mpc(0.5, im_mid)
        z_mid = zeta_phi(s_mid, phi, n_max=2000, precision=25, adaptive=False)

        if abs(im2 - im1) < tol:
            break

        if use_real:
            if z1.real * z_mid.real < 0:
                im2 = im_mid
                z2 = z_mid
            else:
                im1 = im_mid
                z1 = z_mid
        else:
            if z1.imag * z_mid.imag < 0:
                im2 = im_mid
                z2 = z_mid
            else:
                im1 = im_mid
                z1 = z_mid

    im_final = (im1 + im2) / 2
    s_final = mpc(0.5, im_final)
    z_final = zeta_phi(s_final, phi, n_max=3000, precision=30, adaptive=False)

    return im_final, float(fabs(z_final))


def main():
    print("=" * 70)
    print("FAST ZERO SEARCH")
    print("=" * 70)

    # Test Identity first - should have zeros at Riemann zero locations / 2
    # because zeta_phi(s) = zeta(2s) for identity
    print("\n" + "=" * 70)
    print("Testing: Identity (phi(n) = n)")
    print("Note: zeta_phi(s) = sum n^(-s)/n^s = sum n^(-2s) = zeta(2s)")
    print("Zeros should be at Im(s) = Im(rho)/2 where rho is Riemann zero")
    print("=" * 70)

    phi = IdentityTransfer()
    phi.precision = 25

    # Riemann zeros: 14.134725, 21.022040, 25.010858, ...
    # So we expect zeros at: 7.067, 10.511, 12.505, ...

    print("\nChecking at expected zero locations (Riemann Im / 2):")
    expected = [14.134725/2, 21.022040/2, 25.010858/2, 30.424876/2]

    for rz_half in expected:
        s = mpc(0.5, rz_half)
        z = zeta_phi(s, phi, n_max=2000, precision=25, adaptive=False)
        print(f"  Im = {rz_half:.6f}: |zeta| = {float(fabs(z)):.6e}")

    # Scan for sign changes
    print("\nScanning for sign changes...")
    crossings = quick_scan(phi, im_min=5, im_max=20, step=0.1)

    if crossings:
        print(f"Found {len(crossings)} zero crossing(s):")
        for im1, im2, part in crossings[:10]:
            print(f"  Between Im = {im1:.2f} and {im2:.2f} ({part} part)")

            # Bisect to refine
            im_zero, mag_zero = bisect_zero(phi, im1, im2)
            print(f"    -> Refined: Im = {im_zero:.10f}, |zeta| = {mag_zero:.2e}")

            # Compare to Riemann zero / 2
            riemann_half = [14.134725/2, 21.022040/2, 25.010858/2, 30.424876/2, 32.935062/2]
            nearest = min(riemann_half, key=lambda r: abs(r - im_zero))
            print(f"    -> Expected (Riemann/2): {nearest:.6f}, diff = {im_zero - nearest:+.6f}")

    # Now test other arithmetics
    print("\n" + "=" * 70)
    print("Testing other arithmetics:")
    print("=" * 70)

    other_arithmetics = [
        ("Affine_2n (phi=2n)", AffineTransfer(2, 0)),
        ("CoherentTwist_double", CoherentTwistTransfer(lambda x: 2*x, lambda y: y/2, "double")),
        ("Exponential_1.1", ExponentialTransfer(1.1)),
    ]

    for name, phi in other_arithmetics:
        print(f"\n{name}:")
        phi.precision = 20

        crossings = quick_scan(phi, im_min=5, im_max=25, step=0.2)

        if crossings:
            print(f"  Found {len(crossings)} crossing(s)")
            for im1, im2, part in crossings[:3]:
                im_zero, mag_zero = bisect_zero(phi, im1, im2)
                status = "ZERO!" if mag_zero < 1e-8 else ("likely" if mag_zero < 1e-4 else "minimum")
                print(f"    Im = {im_zero:.8f}, |zeta| = {mag_zero:.2e} [{status}]")
        else:
            print(f"  No crossings found in [5, 25]")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

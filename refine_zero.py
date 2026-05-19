"""
Refine the zero candidate for Hierarchy_base10_level2 with high precision.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, diff, nstr
from arithmetics.core.transfer import IteratedExponentialTransfer
from arithmetics.zeta.dirichlet import zeta_phi

def refine_zero_newton(phi, s0, precision=50, max_iter=100, tol=None):
    """
    Refine a zero using Newton-Raphson iteration.

    Args:
        phi: ArithmeticTransfer
        s0: Initial guess (complex)
        precision: Decimal places
        max_iter: Maximum iterations
        tol: Convergence tolerance (default: 10^(-precision+5))

    Returns:
        Refined zero location, final |zeta| value, iterations used
    """
    mp.dps = precision + 10
    phi.precision = precision + 10

    if tol is None:
        tol = mp.mpf(10) ** (-(precision - 5))

    s = mpc(s0)
    n_max = 5000  # More terms for higher precision

    print(f"Starting Newton-Raphson refinement")
    print(f"  Initial: s = {nstr(s.real, 8)} + {nstr(s.imag, 8)}i")
    print(f"  Precision: {precision} decimal places")
    print(f"  Tolerance: {tol}")
    print()

    for iteration in range(max_iter):
        # Evaluate zeta at current point
        z = zeta_phi(s, phi, n_max=n_max, precision=precision+10, adaptive=True)
        mag = fabs(z)

        # Numerical derivative using central difference
        h = mp.mpf(10) ** (-precision // 2)
        z_plus = zeta_phi(s + h, phi, n_max=n_max, precision=precision+10, adaptive=True)
        z_minus = zeta_phi(s - h, phi, n_max=n_max, precision=precision+10, adaptive=True)
        dz = (z_plus - z_minus) / (2 * h)

        # Also compute derivative in imaginary direction
        z_plus_i = zeta_phi(s + h*1j, phi, n_max=n_max, precision=precision+10, adaptive=True)
        z_minus_i = zeta_phi(s - h*1j, phi, n_max=n_max, precision=precision+10, adaptive=True)
        dz_i = (z_plus_i - z_minus_i) / (2 * h * 1j)

        # Use the derivative with larger magnitude for stability
        deriv = dz if fabs(dz) > fabs(dz_i) else dz_i

        print(f"  Iter {iteration+1:3d}: |zeta| = {nstr(mag, 6)}, s = {nstr(s.real, 12)} + {nstr(s.imag, 12)}i")

        # Check convergence
        if mag < tol:
            print(f"\nConverged after {iteration+1} iterations!")
            return s, mag, iteration + 1

        # Newton step
        if fabs(deriv) < mp.mpf(10) ** (-precision):
            print("  Warning: derivative too small, using smaller step")
            step = z / (deriv + mp.mpf(10) ** (-precision // 2))
        else:
            step = z / deriv

        # Damping for stability
        step_mag = fabs(step)
        if step_mag > 1.0:
            step = step / step_mag  # Limit step size

        s_new = s - step

        # Check if we're making progress
        z_new = zeta_phi(s_new, phi, n_max=n_max, precision=precision+10, adaptive=True)
        if fabs(z_new) >= mag:
            # Reduce step size if not improving
            s_new = s - step * 0.5

        s = s_new

    print(f"\nDid not converge after {max_iter} iterations")
    return s, fabs(zeta_phi(s, phi, n_max=n_max, precision=precision+10, adaptive=True)), max_iter


def grid_search_refine(phi, s0, precision=30, grid_size=0.01, n_points=21):
    """
    Fine grid search around initial guess to find better starting point.
    """
    mp.dps = precision + 5
    phi.precision = precision + 5

    re0, im0 = float(s0.real), float(s0.imag)

    best_s = s0
    best_mag = float('inf')

    print(f"Grid search around s = {re0} + {im0}i")
    print(f"  Grid: {n_points}x{n_points}, step = {grid_size}")

    for i in range(n_points):
        for j in range(n_points):
            re = re0 + (i - n_points//2) * grid_size
            im = im0 + (j - n_points//2) * grid_size
            s = mpc(re, im)

            try:
                z = zeta_phi(s, phi, n_max=3000, precision=precision, adaptive=True)
                mag = float(fabs(z))

                if mag < best_mag:
                    best_mag = mag
                    best_s = s
            except:
                pass

    print(f"  Best found: s = {nstr(best_s.real, 8)} + {nstr(best_s.imag, 8)}i, |zeta| = {best_mag:.6e}")
    return best_s, best_mag


def main():
    print("=" * 70)
    print("ZERO REFINEMENT: Hierarchy_base10_level2")
    print("=" * 70)
    print()

    # Create the arithmetic transfer
    phi = IteratedExponentialTransfer(base=10, level=2)
    print(f"Arithmetic: {phi.name}")
    print(f"  phi(n) = 10^(10^n) for level 2")
    print()

    # Initial candidate from survey
    s0 = mpc(0.5, 14.134725)  # Near first Riemann zero

    # Phase 1: Grid search to find better starting point
    print("-" * 70)
    print("PHASE 1: Fine Grid Search")
    print("-" * 70)
    s1, mag1 = grid_search_refine(phi, s0, precision=20, grid_size=0.005, n_points=21)
    print()

    # Phase 2: Finer grid search
    print("-" * 70)
    print("PHASE 2: Ultra-fine Grid Search")
    print("-" * 70)
    s2, mag2 = grid_search_refine(phi, s1, precision=25, grid_size=0.001, n_points=21)
    print()

    # Phase 3: Newton-Raphson refinement at moderate precision
    print("-" * 70)
    print("PHASE 3: Newton-Raphson (30 digits)")
    print("-" * 70)
    s3, mag3, iters3 = refine_zero_newton(phi, s2, precision=30, max_iter=50)
    print()

    # Phase 4: High precision refinement
    print("-" * 70)
    print("PHASE 4: High Precision Refinement (50 digits)")
    print("-" * 70)
    s4, mag4, iters4 = refine_zero_newton(phi, s3, precision=50, max_iter=50)
    print()

    # Final result
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    mp.dps = 50
    print(f"Arithmetic: {phi.name}")
    print()
    print(f"Zero location:")
    print(f"  Re(s) = {nstr(s4.real, 40)}")
    print(f"  Im(s) = {nstr(s4.imag, 40)}")
    print()
    print(f"Verification:")
    print(f"  |zeta_phi(s)| = {nstr(mag4, 10)}")
    print()

    # Compare with first Riemann zero
    riemann_zero_1 = mp.mpf("14.134725141734693790457251983562470270784257115699")
    print(f"Comparison with first Riemann zero:")
    print(f"  Riemann Im = {nstr(riemann_zero_1, 40)}")
    print(f"  Difference = {nstr(abs(s4.imag - riemann_zero_1), 10)}")
    print()

    # Check if on critical line
    print(f"Critical line check:")
    print(f"  |Re(s) - 0.5| = {nstr(abs(s4.real - mp.mpf('0.5')), 10)}")

    return s4, mag4


if __name__ == "__main__":
    result = main()

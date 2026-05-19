"""
Diagnose the zeta computation for Hierarchy_base10_level2.
"""

import sys
sys.path.insert(0, r"C:\Users\hp\desktop\claude\os\simulation")

from mpmath import mp, mpc, fabs, log10, power, nstr, exp, ln
from arithmetics.core.transfer import IteratedExponentialTransfer

def diagnose_series():
    """
    Check term-by-term behavior of the Dirichlet series.
    """
    mp.dps = 50

    phi = IteratedExponentialTransfer(base=10, level=2)
    print("=" * 70)
    print("DIAGNOSTIC: Hierarchy_base10_level2 Dirichlet Series")
    print("=" * 70)
    print()

    print(f"Arithmetic: {phi.name}")
    print(f"  phi(n) = 10^(10^n)")
    print()

    # Show phi values for small n
    print("phi(n) values:")
    print("-" * 50)
    for n in [1, 2, 3, 4, 5]:
        phi_n = phi.phi(n)
        log_phi = float(log10(phi_n)) if phi_n > 0 else 'undefined'
        print(f"  phi({n}) = 10^(10^{n}) = 10^{10**n}")
        print(f"         ~= 10^{log_phi:.2e}")
    print()

    # The issue: phi(1) = 10^10 which is already huge
    # phi(n)^(-s) for Re(s) > 0 becomes astronomically small

    print("Term magnitudes for s = 0.5 + 14.1i:")
    print("-" * 50)

    s = mpc(0.5, 14.1347)
    print(f"  s = {s}")
    print()

    # Term: phi(n)^(-s) / n^s
    # = 10^(-10^n * s) / n^s
    # log10(|term|) = -10^n * Re(s) - log10(n) * Re(s)

    for n in [1, 2, 3]:
        phi_n = phi.phi(n)

        # phi(n)^(-s) = exp(-s * ln(phi(n)))
        # For phi(n) = 10^(10^n), ln(phi(n)) = 10^n * ln(10)

        ln_phi_n = (mp.mpf(10) ** n) * ln(10)
        log10_term_magnitude = -float(s.real * ln_phi_n / ln(10)) - float(s.real * log10(n))

        print(f"  n = {n}:")
        print(f"    phi({n}) = 10^{10**n}")
        print(f"    log10(|term|) ~= {log10_term_magnitude:.2e}")

        if n == 1:
            # Actually compute for n=1
            term = power(phi_n, -s) / power(n, s)
            print(f"    Actual |term| = {nstr(fabs(term), 6)}")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("For Hierarchy_base10_level2:")
    print("  phi(1) = 10^10 (10 billion)")
    print("  phi(2) = 10^100 (googol)")
    print("  phi(3) = 10^1000 (googolplex-ish)")
    print()
    print("The Dirichlet series is COMPLETELY dominated by the n=1 term.")
    print("All other terms are negligibly small (< 10^(-50) relative to n=1).")
    print()
    print("This means zeta_phi(s) ~= phi(1)^(-s) = 10^(-10*s)")
    print()
    print("For s = 0.5 + ti:")
    print("  zeta_phi(s) ~= 10^(-10*(0.5 + ti)) = 10^(-5) * 10^(-10ti)")
    print("  |zeta_phi(s)| ~= 10^(-5) regardless of Im(s)")
    print()
    print("This is NOT a zero - it's just a very small constant!")
    print()

    # Let's try a more interesting arithmetic
    print("=" * 70)
    print("COMPARISON: Trying lower hierarchy level")
    print("=" * 70)
    print()

    # Try level 0 (phi(n) = n) - this is identity
    # Try level 1 (phi(n) = base^n)

    for level in [0, 1]:
        phi_test = IteratedExponentialTransfer(base=10, level=level)
        print(f"\nHierarchy_base10_level{level}: phi(n) = ", end="")
        if level == 0:
            print("n")
        elif level == 1:
            print("10^n")
        elif level == 2:
            print("10^(10^n)")

        # Check some values
        s_test = mpc(0.5, 14.1347)
        terms = []
        for n in range(1, 6):
            phi_n = phi_test.phi(n)
            term = power(phi_n, -s_test) / power(n, s_test)
            mag = float(fabs(term))
            terms.append((n, mag))
            print(f"    n={n}: |term| = {mag:.6e}")

        # Sum first 100 terms
        total = mpc(0, 0)
        for n in range(1, 101):
            phi_n = phi_test.phi(n)
            term = power(phi_n, -s_test) / power(n, s_test)
            total += term

        print(f"    Sum of first 100 terms: |zeta| = {float(fabs(total)):.6e}")


if __name__ == "__main__":
    diagnose_series()

"""
Test rapide de la somme interne D_p(s) = sum (log^{o p}(n))^{-(s+1)}
Version optimisee pour execution rapide.
"""

import numpy as np
from mpmath import mp, mpf, mpc, exp, log, power, fabs, isnan, isinf
from datetime import datetime
from pathlib import Path
import json

mp.dps = 30  # Precision reduite pour rapidite

print("=" * 70)
print("TEST RAPIDE - ZEROS DE D_p(s) = sum (log^p(n))^{-(s+1)}")
print("=" * 70)


def log_iterated(x, p):
    x = mpf(x)
    if p == 0:
        return x
    result = x
    for _ in range(p):
        if result <= 0:
            return mpf('nan')
        result = log(result)
    return result


def inner_sum(s, p, N=200):
    """Somme interne D_p(s)."""
    s = mpc(s)
    total = mpc(0)
    n_min = 2 if p < 2 else (3 if p < 3 else 16)

    for n in range(n_min, N + n_min):
        log_p_n = log_iterated(mpf(n), p)
        if isnan(log_p_n) or log_p_n <= 0:
            continue
        term = power(log_p_n, -(s + 1))
        if not (isnan(term) or isinf(fabs(term))):
            total += term
    return total


def find_zero(p, s_init, max_iter=50):
    """Newton-Raphson simplifie."""
    s = mpc(s_init)
    h = mpf('1e-5')

    for _ in range(max_iter):
        D = inner_sum(s, p)
        if fabs(D) < 1e-12:
            return float(s.real), float(s.imag), float(fabs(D)), True

        D_plus = inner_sum(s + h, p)
        dD = (D_plus - D) / h

        if fabs(dD) < 1e-20:
            s += mpc(0.05, 0.1)
            continue

        step = D / dD
        if fabs(step) > 0.2:
            step *= 0.2 / fabs(step)

        s -= step

        # Bornes
        if s.real < 0:
            s = mpc(0.1, s.imag)
        if s.real > 1.5:
            s = mpc(1.4, s.imag)
        if s.imag < 1:
            s = mpc(s.real, 5)

    return float(s.real), float(s.imag), float(fabs(inner_sum(s, p))), False


# Test rapide pour chaque niveau
results = {}

for p in [0, 1, 2, 3]:
    print(f"\n{'='*50}")
    print(f"NIVEAU p = {p}")
    print(f"{'='*50}")

    zeros = []

    # Points de depart bases sur zeros de Riemann
    inits = [
        (0.5, 14.13), (0.5, 21.02), (0.5, 25.01),
        (0.3, 15), (0.7, 15), (0.5, 30), (0.5, 35)
    ]

    for re_init, im_init in inits:
        re, im, res, conv = find_zero(p, mpc(re_init, im_init))

        if conv or res < 1e-6:
            # Verifier unicite
            is_new = all(abs(re - z[0]) > 0.05 or abs(im - z[1]) > 0.5 for z in zeros)
            if is_new:
                zeros.append((re, im, res))
                dist = abs(re - 0.5)
                status = "LIGNE CRITIQUE" if dist < 0.03 else f"dist={dist:.3f}"
                print(f"  Zero: {re:.4f} + {im:.2f}i  [{status}]")

    results[p] = zeros

    if zeros:
        re_vals = [z[0] for z in zeros]
        print(f"\n  Resume: {len(zeros)} zeros, Re moyen = {np.mean(re_vals):.4f}")


# Synthese
print("\n" + "=" * 70)
print("SYNTHESE")
print("=" * 70)

print(f"\n{'p':<5} {'Zeros':<8} {'Re moyen':<12} {'Sur ligne':<15}")
print("-" * 45)

for p in [0, 1, 2, 3]:
    zeros = results[p]
    if zeros:
        re_vals = [z[0] for z in zeros]
        on_line = sum(1 for r in re_vals if abs(r - 0.5) < 0.05)
        print(f"{p:<5} {len(zeros):<8} {np.mean(re_vals):<12.4f} {on_line}/{len(zeros)}")
    else:
        print(f"{p:<5} 0")


# Interpretation
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Comparer Re(s) entre niveaux
all_re = {p: [z[0] for z in results[p]] for p in results if results[p]}

if all_re:
    print("\nComparaison des Re(s) par niveau:")
    for p, re_vals in all_re.items():
        print(f"  p={p}: Re = {re_vals}")

    # Verifier si constant
    all_means = [np.mean(re_vals) for re_vals in all_re.values()]
    variation = max(all_means) - min(all_means) if all_means else 0

    print(f"\nVariation de Re moyen entre niveaux: {variation:.4f}")

    if variation < 0.1:
        print("\n>>> Les zeros ont Re(s) SIMILAIRE pour tous les niveaux")
        print(">>> Cela suggere une forme d'INVARIANCE structurelle")
    else:
        print("\n>>> Les zeros ont Re(s) DIFFERENTS selon le niveau")
        print(">>> La position des zeros DEPEND des poids log^p(n)")

# Sauvegarde
output = {'results': {str(p): results[p] for p in results}, 'timestamp': str(datetime.now())}
Path("results/inner_sum_tests").mkdir(parents=True, exist_ok=True)
with open("results/inner_sum_tests/fast_results.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResultats sauvegardes.")

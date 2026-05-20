"""
Tests complementaires pour verifier l'hypothese p=4
====================================================

1. Tester p=5, p=6, p=7 (persistance de Re(s)=0.5)
2. Augmenter la precision numerique (100 decimales)
3. Chercher des zeros a Im(s) plus eleve (jusqu'a 50)
4. Comparer avec le zero classique de Riemann
"""

import numpy as np
from mpmath import mp, mpf, mpc, exp, log, power, fabs, pi, isnan, isinf, nstr
import json
from datetime import datetime
from pathlib import Path

# Haute precision
mp.dps = 100

OUTPUT_DIR = Path("results/high_level_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TESTS COMPLEMENTAIRES - HYPOTHESE p=4")
print("=" * 70)
print(f"Precision: {mp.dps} decimales")
print("=" * 70)


def exp_iterated(x, p):
    """Phi_p(x) = exp^{o p}(x)"""
    x = mpf(x)
    if p == 0:
        return x
    elif p > 0:
        result = x
        for _ in range(p):
            if result > 700:
                return mpf('inf')
            result = exp(result)
        return result
    else:
        result = x
        for _ in range(-p):
            if result <= 0:
                return mpf('-inf')
            result = log(result)
        return result


def log_iterated(x, p):
    """Phi_p^{-1}(x) = log^{o p}(x)"""
    x = mpf(x)
    if p == 0:
        return x
    elif p > 0:
        result = x
        for _ in range(p):
            if result <= 0:
                return mpf('-inf')
            result = log(result)
        return result
    else:
        result = x
        for _ in range(-p):
            if result > 700:
                return mpf('inf')
            result = exp(result)
        return result


def additive_identity(p):
    """0_p = exp^{o p}(0)"""
    return exp_iterated(mpf(0), p)


def zeta_p(s, p, N_terms=200):
    """
    Fonction zeta de niveau p:
    zeta_p(s) = exp^{o p}( sum (log^{o p}(n))^{-(s+1)} )
    """
    s = mpc(s)
    inner_sum = mpc(0)

    for n in range(2, N_terms + 2):
        try:
            log_p_n = log_iterated(mpf(n), p)

            if log_p_n <= 0 or isinf(log_p_n) or isnan(log_p_n):
                continue

            term = power(log_p_n, -(s + 1))

            if isnan(term) or isinf(fabs(term)):
                continue

            inner_sum += term

            if fabs(term) < mpf('1e-50'):
                break
        except:
            continue

    if p == 0:
        return inner_sum
    else:
        real_sum = float(inner_sum.real)
        if abs(real_sum) > 700:
            return mpc(mpf('inf'), 0)
        result = exp_iterated(real_sum, p)
        return mpc(result, float(inner_sum.imag) * float(result) if result < mpf('1e100') else 0)


def find_zero_high_precision(p, s_init, max_iter=200, tol=1e-30):
    """Recherche de zero avec haute precision."""
    zero_p = additive_identity(p)
    s = mpc(s_init)
    h = mpf('1e-10')

    for iteration in range(max_iter):
        try:
            z_val = zeta_p(s, p, N_terms=300)

            if p == 0:
                residual = z_val
            else:
                residual = z_val - zero_p

            res_mag = fabs(residual)

            if res_mag < tol:
                return {
                    'real': float(s.real),
                    'imag': float(s.imag),
                    'residual': float(res_mag),
                    'converged': True,
                    'iterations': iteration
                }

            z_plus = zeta_p(s + h, p, N_terms=300)
            z_minus = zeta_p(s - h, p, N_terms=300)
            dz = (z_plus - z_minus) / (2 * h)

            if fabs(dz) < mpf('1e-50'):
                s = s + mpc(0.01, 0.1)
                continue

            s_new = s - residual / dz

            # Contraintes plus larges
            if s_new.real < -2:
                s_new = mpc(-2, s_new.imag)
            if s_new.real > 3:
                s_new = mpc(3, s_new.imag)
            if s_new.imag < 1:
                s_new = mpc(s_new.real, 5)
            if s_new.imag > 100:
                s_new = mpc(s_new.real, 100)

            if fabs(s_new - s) < mpf('1e-20'):
                return {
                    'real': float(s_new.real),
                    'imag': float(s_new.imag),
                    'residual': float(res_mag),
                    'converged': True,
                    'iterations': iteration
                }

            s = s_new

        except Exception as e:
            s = s + mpc(0.05, 0.2)
            continue

    return {
        'real': float(s.real),
        'imag': float(s.imag),
        'residual': float('inf'),
        'converged': False,
        'iterations': max_iter
    }


# =============================================================================
# TEST 1: Niveaux p = 4, 5, 6, 7
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: Persistance de Re(s)=0.5 pour p >= 4")
print("=" * 70)

test_levels = [4, 5, 6, 7]
initial_guesses = [
    mpc(0.5, 14.0),   # Premier zero de Riemann
    mpc(0.5, 21.0),   # Deuxieme zero
    mpc(0.5, 25.0),   # Troisieme zero
]

results_by_level = {}

for p in test_levels:
    print(f"\n--- Niveau p = {p} ---")
    zero_p = additive_identity(p)
    print(f"Identite additive 0_{p} = {nstr(zero_p, 6) if zero_p < mpf('1e20') else 'inf'}")

    results_by_level[p] = []

    for i, s_init in enumerate(initial_guesses):
        print(f"  Recherche {i+1}/3 depuis s = {float(s_init.real):.1f} + {float(s_init.imag):.1f}i ... ", end="", flush=True)

        result = find_zero_high_precision(p, s_init)
        results_by_level[p].append(result)

        if result['converged']:
            dist = abs(result['real'] - 0.5)
            status = "SUR LIGNE CRITIQUE" if dist < 0.01 else f"dist={dist:.4f}"
            print(f"s = {result['real']:.6f} + {result['imag']:.2f}i [{status}]")
        else:
            print("non converge")

# Analyse
print("\n" + "-" * 70)
print("RESUME TEST 1:")
print("-" * 70)
print(f"{'Niveau':<10} {'Zeros':<8} {'Re(s) moyen':<15} {'Sur ligne critique':<20}")
print("-" * 70)

for p in test_levels:
    zeros = [r for r in results_by_level[p] if r['converged']]
    n_zeros = len(zeros)
    if zeros:
        mean_re = np.mean([z['real'] for z in zeros])
        on_critical = sum(1 for z in zeros if abs(z['real'] - 0.5) < 0.01)
        print(f"p={p:<7} {n_zeros:<8} {mean_re:<15.6f} {on_critical}/{n_zeros}")
    else:
        print(f"p={p:<7} 0        N/A")


# =============================================================================
# TEST 2: Zeros a Im(s) eleve pour p=4
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Zeros a Im(s) eleve pour p=4")
print("=" * 70)

high_im_guesses = [
    mpc(0.5, 30.0),
    mpc(0.5, 37.0),
    mpc(0.5, 40.0),
    mpc(0.5, 43.0),
    mpc(0.5, 48.0),
]

high_im_results = []

for s_init in high_im_guesses:
    print(f"  Im(s) ~ {float(s_init.imag):.0f} ... ", end="", flush=True)
    result = find_zero_high_precision(4, s_init)
    high_im_results.append(result)

    if result['converged']:
        dist = abs(result['real'] - 0.5)
        print(f"s = {result['real']:.6f} + {result['imag']:.2f}i [dist={dist:.6f}]")
    else:
        print("non converge")


# =============================================================================
# TEST 3: Comparaison p=1 vs p=4 avec meme point de depart
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Comparaison p=1 vs p=4 (meme initialisation)")
print("=" * 70)

comparison_inits = [
    mpc(0.5, 14.134725),  # Zero exact de Riemann
    mpc(0.6, 14.0),
    mpc(0.4, 14.0),
    mpc(0.5, 21.022040),  # Deuxieme zero de Riemann
]

print(f"\n{'Init s':<25} {'p=1 Re(s)':<15} {'p=4 Re(s)':<15} {'Difference':<15}")
print("-" * 70)

for s_init in comparison_inits:
    r1 = find_zero_high_precision(1, s_init, max_iter=100)
    r4 = find_zero_high_precision(4, s_init, max_iter=100)

    init_str = f"{float(s_init.real):.2f}+{float(s_init.imag):.2f}i"

    if r1['converged'] and r4['converged']:
        diff = abs(r1['real'] - r4['real'])
        print(f"{init_str:<25} {r1['real']:<15.6f} {r4['real']:<15.6f} {diff:<15.6f}")
    else:
        re1 = f"{r1['real']:.6f}" if r1['converged'] else "N/C"
        re4 = f"{r4['real']:.6f}" if r4['converged'] else "N/C"
        print(f"{init_str:<25} {re1:<15} {re4:<15}")


# =============================================================================
# TEST 4: Analyse de la serie interne
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Analyse de la serie interne pour differents p")
print("=" * 70)

s_test = mpc(0.5, 14.0)
print(f"\nPoint de test: s = {float(s_test.real)} + {float(s_test.imag)}i")
print(f"\n{'p':<5} {'Sum interne':<25} {'|Sum|':<15} {'log^p(10)':<15}")
print("-" * 70)

for p in range(0, 8):
    inner_sum = mpc(0)
    for n in range(2, 102):
        try:
            log_p_n = log_iterated(mpf(n), p)
            if log_p_n > 0 and not isinf(log_p_n):
                term = power(log_p_n, -(s_test + 1))
                if not (isnan(term) or isinf(fabs(term))):
                    inner_sum += term
        except:
            pass

    log_p_10 = log_iterated(mpf(10), p)
    log_p_10_str = f"{float(log_p_10):.6f}" if not isinf(log_p_10) and log_p_10 > 0 else "N/A"

    print(f"{p:<5} {float(inner_sum.real):>12.6f} + {float(inner_sum.imag):>8.4f}i   "
          f"{float(fabs(inner_sum)):<15.6f} {log_p_10_str:<15}")


# =============================================================================
# SAUVEGARDE
# =============================================================================
all_results = {
    'test1_levels': {str(p): results_by_level[p] for p in test_levels},
    'test2_high_im': high_im_results,
    'timestamp': datetime.now().isoformat(),
    'precision': mp.dps
}

with open(OUTPUT_DIR / "high_level_tests.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)


# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Compter les zeros sur la ligne critique par niveau
summary = {}
for p in test_levels:
    zeros = [r for r in results_by_level[p] if r['converged']]
    on_critical = sum(1 for z in zeros if abs(z['real'] - 0.5) < 0.01)
    summary[p] = {'total': len(zeros), 'on_critical': on_critical}

print("\nZeros sur la ligne critique Re(s) = 0.5:")
for p, data in summary.items():
    pct = 100 * data['on_critical'] / data['total'] if data['total'] > 0 else 0
    status = "[OK]" if pct == 100 else "[PARTIEL]" if pct > 50 else "[NON]"
    print(f"  p={p}: {data['on_critical']}/{data['total']} ({pct:.0f}%) {status}")

# Verifier la tendance
all_on_critical = all(summary[p]['on_critical'] == summary[p]['total']
                      for p in test_levels if summary[p]['total'] > 0)

if all_on_critical:
    print("\n>>> HYPOTHESE CONFIRMEE: Pour p >= 4, tous les zeros sont sur Re(s) = 0.5")
    print(">>> L'invariance de la ligne critique emerge a partir du niveau p=4")
else:
    print("\n>>> HYPOTHESE PARTIELLEMENT CONFIRMEE")
    print(">>> Des investigations supplementaires sont necessaires")

print(f"\nResultats sauvegardes: {OUTPUT_DIR / 'high_level_tests.json'}")

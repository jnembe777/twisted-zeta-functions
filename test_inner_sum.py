"""
Test de la somme interne D_p(s) = sum (log^{o p}(n))^{-(s+1)}

Selon le theoreme d'invariance (correction_bis.pdf):
- zeta_p(s) = 0_p <=> D_p(s) = 0 (par injectivite de exp^{o p})
- Les zeros de D_p(s) sont les VRAIS zeros a etudier
- La conjugaison agit sur les VALEURS, pas sur s
- Donc les zeros dans le plan s sont structures par D_p(s)
"""

import numpy as np
from mpmath import mp, mpf, mpc, exp, log, power, fabs, isnan, isinf, nstr
import json
from datetime import datetime
from pathlib import Path

mp.dps = 50

OUTPUT_DIR = Path("results/inner_sum_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TEST DES ZEROS DE LA SOMME INTERNE D_p(s)")
print("D_p(s) = sum (log^{o p}(n))^{-(s+1)}")
print("=" * 70)


def log_iterated(x, p):
    """log^{o p}(x)"""
    x = mpf(x)
    if p == 0:
        return x
    elif p > 0:
        result = x
        for _ in range(p):
            if result <= 0:
                return mpf('nan')
            result = log(result)
        return result
    else:
        result = x
        for _ in range(-p):
            if result > 700:
                return mpf('inf')
            result = exp(result)
        return result


def inner_sum_Dp(s, p, N_terms=1000):
    """
    Calcule D_p(s) = sum_{n>=n_min} (log^{o p}(n))^{-(s+1)}

    C'est la somme INTERNE avant application de exp^{o p}.
    Les zeros de cette fonction sont les vrais zeros a etudier.
    """
    s = mpc(s)
    total = mpc(0)
    n_valid = 0

    # n_min depend de p (log^p(n) doit etre > 0)
    # p=0: n >= 1
    # p=1: n >= 2 (log(n) > 0)
    # p=2: n >= 3 (log(log(n)) > 0 pour n > e)
    # p=3: n >= 16 (log^3(n) > 0 pour n > e^e ~ 15.15)

    n_min = 2
    if p >= 2:
        n_min = 3
    if p >= 3:
        n_min = 16

    for n in range(n_min, N_terms + n_min):
        try:
            log_p_n = log_iterated(mpf(n), p)

            if isnan(log_p_n) or log_p_n <= 0:
                continue

            # (log^p(n))^{-(s+1)}
            term = power(log_p_n, -(s + 1))

            if isnan(term) or isinf(fabs(term)):
                continue

            total += term
            n_valid += 1

            # Convergence
            if n_valid > 50 and fabs(term) < mpf('1e-40'):
                break

        except:
            continue

    return total, n_valid


def find_zero_inner_sum(p, s_init, max_iter=200, tol=1e-20):
    """Trouve un zero de D_p(s)."""
    s = mpc(s_init)
    h = mpf('1e-7')

    for iteration in range(max_iter):
        try:
            Dp, n_terms = inner_sum_Dp(s, p)
            mag = fabs(Dp)

            if mag < tol:
                return {
                    'real': float(s.real),
                    'imag': float(s.imag),
                    'residual': float(mag),
                    'converged': True,
                    'iterations': iteration,
                    'n_terms': n_terms
                }

            # Derivee numerique
            Dp_plus, _ = inner_sum_Dp(s + h, p)
            Dp_minus, _ = inner_sum_Dp(s - h, p)
            dDp = (Dp_plus - Dp_minus) / (2 * h)

            if fabs(dDp) < mpf('1e-35'):
                s = s + mpc(0.02, 0.15)
                continue

            step = Dp / dDp
            step_mag = fabs(step)
            if step_mag > 0.3:
                step = step * (0.3 / step_mag)

            s_new = s - step

            # Contraintes
            if s_new.real < -2:
                s_new = mpc(-2, s_new.imag)
            if s_new.real > 3:
                s_new = mpc(3, s_new.imag)
            if s_new.imag < 0.5:
                s_new = mpc(s_new.real, 5)

            if fabs(s_new - s) < mpf('1e-15'):
                Dp_final, _ = inner_sum_Dp(s_new, p)
                return {
                    'real': float(s_new.real),
                    'imag': float(s_new.imag),
                    'residual': float(fabs(Dp_final)),
                    'converged': fabs(Dp_final) < 1e-8,
                    'iterations': iteration,
                    'n_terms': n_terms
                }

            s = s_new

        except:
            s = s + mpc(0.03, 0.1)

    Dp_final, n_terms = inner_sum_Dp(s, p)
    return {
        'real': float(s.real),
        'imag': float(s.imag),
        'residual': float(fabs(Dp_final)) if fabs(Dp_final) < mpf('1e50') else float('inf'),
        'converged': False,
        'iterations': max_iter,
        'n_terms': n_terms
    }


def grid_search_inner(p, re_range, im_range, re_step=0.05, im_step=0.5):
    """Recherche sur grille pour D_p(s) = 0."""
    candidates = []

    re_vals = np.arange(re_range[0], re_range[1] + re_step, re_step)
    im_vals = np.arange(im_range[0], im_range[1] + im_step, im_step)

    for re_val in re_vals:
        for im_val in im_vals:
            s = mpc(re_val, im_val)
            try:
                Dp, _ = inner_sum_Dp(s, p, N_terms=300)
                mag = float(fabs(Dp))
                if mag < 2.0:
                    candidates.append({'re': re_val, 'im': im_val, 'mag': mag})
            except:
                pass

    candidates.sort(key=lambda x: x['mag'])
    return candidates[:30]


# =============================================================================
# Tests pour chaque niveau
# =============================================================================

results_all = {}

for p in [0, 1, 2, 3]:
    print(f"\n{'=' * 70}")
    print(f"NIVEAU p = {p}")
    print(f"{'=' * 70}")

    # Info sur les termes valides
    n_min = 2 if p < 2 else (3 if p < 3 else 16)
    print(f"Premier terme valide: n = {n_min}")

    # Quelques valeurs de log^p(n)
    print(f"\nValeurs de log^{p}(n):")
    for n in [n_min, n_min+5, 50, 100]:
        val = log_iterated(mpf(n), p)
        if not isnan(val) and val > 0:
            print(f"  n={n}: log^{p}(n) = {float(val):.6f}")

    # Recherche sur grille
    print(f"\nRecherche sur grille...")
    candidates = grid_search_inner(p, (0, 1.5), (5, 50), re_step=0.1, im_step=1.0)
    print(f"  {len(candidates)} candidats trouves")

    if candidates:
        print(f"\n  Top 5 candidats (|D_p(s)|):")
        for c in candidates[:5]:
            print(f"    s = {c['re']:.2f} + {c['im']:.1f}i, |D_p| = {c['mag']:.6f}")

    # Raffiner les zeros
    print(f"\nRaffinement des zeros de D_{p}(s)...")
    zeros = []

    # Points de depart: candidats + points standards
    inits = [mpc(c['re'], c['im']) for c in candidates[:15]]
    inits.extend([mpc(0.5, t) for t in [14.13, 21.02, 25.01, 30.42, 32.94, 37.59, 40.92]])

    seen = set()
    for s_init in inits:
        key = (round(float(s_init.real), 1), round(float(s_init.imag), 0))
        if key in seen:
            continue
        seen.add(key)

        result = find_zero_inner_sum(p, s_init)

        if result['converged'] or result['residual'] < 1e-6:
            # Verifier unicite
            is_new = True
            for prev in zeros:
                if abs(prev['real'] - result['real']) < 0.02 and abs(prev['imag'] - result['imag']) < 0.3:
                    is_new = False
                    break
            if is_new:
                zeros.append(result)
                dist = abs(result['real'] - 0.5)
                status = "SUR 0.5" if dist < 0.02 else f"dist={dist:.4f}"
                print(f"  Zero: s = {result['real']:.6f} + {result['imag']:.2f}i [{status}]")

    results_all[p] = zeros

    # Stats
    if zeros:
        re_vals = [z['real'] for z in zeros]
        print(f"\n  Statistiques:")
        print(f"    Nombre de zeros: {len(zeros)}")
        print(f"    Re(s) moyen: {np.mean(re_vals):.6f}")
        print(f"    Re(s) min/max: {min(re_vals):.4f} / {max(re_vals):.4f}")
        print(f"    Distance moyenne a 0.5: {np.mean([abs(r - 0.5) for r in re_vals]):.6f}")


# =============================================================================
# Comparaison finale
# =============================================================================
print("\n" + "=" * 70)
print("COMPARAISON FINALE - ZEROS DE D_p(s)")
print("=" * 70)

print(f"\n{'p':<5} {'Zeros':<8} {'Re moyen':<12} {'Dist a 0.5':<12} {'Sur ligne':<12}")
print("-" * 55)

for p in [0, 1, 2, 3]:
    zeros = results_all[p]
    if zeros:
        re_vals = [z['real'] for z in zeros]
        mean_re = np.mean(re_vals)
        mean_dist = np.mean([abs(r - 0.5) for r in re_vals])
        on_line = sum(1 for r in re_vals if abs(r - 0.5) < 0.05)
        print(f"{p:<5} {len(zeros):<8} {mean_re:<12.4f} {mean_dist:<12.4f} {on_line}/{len(zeros)}")
    else:
        print(f"{p:<5} 0")


# =============================================================================
# Test d'invariance: memes zeros pour differents p?
# =============================================================================
print("\n" + "=" * 70)
print("TEST D'INVARIANCE: Position des premiers zeros")
print("=" * 70)

# Comparer les positions Im(s) des premiers zeros
print("\nPremiers zeros par niveau (tries par Im(s)):")
for p in [0, 1, 2, 3]:
    zeros = sorted(results_all[p], key=lambda z: z['imag'])[:3]
    if zeros:
        print(f"  p={p}: ", end="")
        for z in zeros:
            print(f"{z['real']:.3f}+{z['imag']:.2f}i  ", end="")
        print()


# =============================================================================
# Conclusion
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION SUR L'INVARIANCE")
print("=" * 70)

# Verifier si Re(s) est constant a 0.5 pour tous les niveaux
all_on_critical = True
for p in [0, 1, 2, 3]:
    zeros = results_all[p]
    if zeros:
        for z in zeros:
            if abs(z['real'] - 0.5) > 0.1:
                all_on_critical = False
                break

if all_on_critical:
    print("\n>>> INVARIANCE CONFIRMEE: Tous les zeros ont Re(s) ~ 0.5")
else:
    print("\n>>> INVARIANCE NON CONFIRMEE pour cette serie")
    print("    Les zeros de D_p(s) = sum (log^p(n))^{-(s+1)} ont des")
    print("    parties reelles qui varient avec p.")
    print("\n    NOTE: Ceci est ATTENDU car D_p(s) n'est PAS la fonction")
    print("    zeta de Riemann classique. Les poids log^p(n) modifient")
    print("    la structure analytique de la serie.")

# Sauvegarder
output_data = {
    'results': {str(p): results_all[p] for p in results_all},
    'timestamp': datetime.now().isoformat()
}

with open(OUTPUT_DIR / "inner_sum_zeros.json", 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

print(f"\nResultats sauvegardes: {OUTPUT_DIR / 'inner_sum_zeros.json'}")

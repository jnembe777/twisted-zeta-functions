"""
Tests approfondis pour p=2 et p=3
=================================

Ces niveaux ont des series non-triviales et permettent une vraie
verification de l'invariance de la ligne critique.
"""

import numpy as np
from mpmath import mp, mpf, mpc, exp, log, power, fabs, pi, isnan, isinf, nstr, re, im
import json
from datetime import datetime
from pathlib import Path

# Haute precision
mp.dps = 80

OUTPUT_DIR = Path("results/p2_p3_deep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TESTS APPROFONDIS - NIVEAUX p=2 et p=3")
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


def zeta_p_detailed(s, p, N_terms=500):
    """
    Fonction zeta de niveau p avec details.
    Retourne (valeur, somme_interne, n_termes_valides)
    """
    s = mpc(s)
    inner_sum = mpc(0)
    n_valid = 0

    for n in range(2, N_terms + 2):
        try:
            log_p_n = log_iterated(mpf(n), p)

            if log_p_n <= 0 or isinf(log_p_n) or isnan(log_p_n):
                continue

            term = power(log_p_n, -(s + 1))

            if isnan(term) or isinf(fabs(term)):
                continue

            inner_sum += term
            n_valid += 1

            if fabs(term) < mpf('1e-60'):
                break
        except:
            continue

    if p == 0:
        return inner_sum, inner_sum, n_valid
    else:
        real_sum = float(inner_sum.real)
        if abs(real_sum) > 700:
            return mpc(mpf('inf'), 0), inner_sum, n_valid
        result = exp_iterated(real_sum, p)
        return mpc(result, float(inner_sum.imag) * float(result) if result < mpf('1e100') else 0), inner_sum, n_valid


def zeta_p(s, p, N_terms=500):
    """Version simple."""
    val, _, _ = zeta_p_detailed(s, p, N_terms)
    return val


def find_zero_precise(p, s_init, max_iter=300, tol=1e-25):
    """Recherche de zero precise avec suivi."""
    zero_p = additive_identity(p)
    s = mpc(s_init)
    h = mpf('1e-8')

    trajectory = []

    for iteration in range(max_iter):
        try:
            z_val, inner_sum, n_terms = zeta_p_detailed(s, p)

            if p == 0:
                residual = z_val
            else:
                residual = z_val - zero_p

            res_mag = fabs(residual)

            trajectory.append({
                'iter': iteration,
                'real': float(s.real),
                'imag': float(s.imag),
                'residual': float(res_mag) if res_mag < mpf('1e50') else float('inf')
            })

            if res_mag < tol:
                return {
                    'real': float(s.real),
                    'imag': float(s.imag),
                    'residual': float(res_mag),
                    'converged': True,
                    'iterations': iteration,
                    'inner_sum_mag': float(fabs(inner_sum)),
                    'n_terms': n_terms,
                    'trajectory': trajectory[-5:]  # Derniers 5 points
                }

            # Derivee numerique
            z_plus = zeta_p(s + h, p)
            z_minus = zeta_p(s - h, p)
            dz = (z_plus - z_minus) / (2 * h)

            if fabs(dz) < mpf('1e-40'):
                s = s + mpc(0.02, 0.1)
                continue

            step = residual / dz

            # Limiter le pas
            step_mag = fabs(step)
            if step_mag > 0.5:
                step = step * (0.5 / step_mag)

            s_new = s - step

            # Contraintes
            if s_new.real < -1:
                s_new = mpc(-1, s_new.imag)
            if s_new.real > 2:
                s_new = mpc(2, s_new.imag)
            if s_new.imag < 0.5:
                s_new = mpc(s_new.real, 5)
            if s_new.imag > 80:
                s_new = mpc(s_new.real, 80)

            if fabs(s_new - s) < mpf('1e-18'):
                return {
                    'real': float(s_new.real),
                    'imag': float(s_new.imag),
                    'residual': float(res_mag) if res_mag < mpf('1e50') else float('inf'),
                    'converged': res_mag < 1e-10,
                    'iterations': iteration,
                    'inner_sum_mag': float(fabs(inner_sum)),
                    'n_terms': n_terms,
                    'trajectory': trajectory[-5:]
                }

            s = s_new

        except Exception as e:
            s = s + mpc(0.03, 0.15)
            continue

    return {
        'real': float(s.real),
        'imag': float(s.imag),
        'residual': float('inf'),
        'converged': False,
        'iterations': max_iter,
        'inner_sum_mag': 0,
        'n_terms': 0,
        'trajectory': trajectory[-5:]
    }


def grid_search(p, re_range, im_range, re_step=0.05, im_step=1.0):
    """Recherche sur grille pour trouver des candidats."""
    candidates = []
    zero_p = additive_identity(p)

    re_vals = np.arange(re_range[0], re_range[1] + re_step, re_step)
    im_vals = np.arange(im_range[0], im_range[1] + im_step, im_step)

    for re_val in re_vals:
        for im_val in im_vals:
            s = mpc(re_val, im_val)
            try:
                z_val = zeta_p(s, p, N_terms=200)
                if p == 0:
                    mag = fabs(z_val)
                else:
                    mag = fabs(z_val - zero_p)

                if mag < 1.0:
                    candidates.append({
                        're': re_val,
                        'im': im_val,
                        'mag': float(mag)
                    })
            except:
                pass

    # Trier par magnitude
    candidates.sort(key=lambda x: x['mag'])
    return candidates[:20]  # Top 20


# =============================================================================
# ANALYSE DES SERIES
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSE DES SERIES INTERNES")
print("=" * 70)

print("\nTermes log^{o p}(n) pour differents n:")
print(f"{'n':<6} {'p=0':<12} {'p=1':<12} {'p=2':<12} {'p=3':<12}")
print("-" * 54)

for n in [2, 3, 5, 10, 20, 50, 100, 200]:
    row = f"{n:<6}"
    for p in [0, 1, 2, 3]:
        val = log_iterated(mpf(n), p)
        if val > 0 and not isinf(val):
            row += f"{float(val):<12.4f}"
        else:
            row += f"{'N/A':<12}"
    print(row)

print("\nNombre de termes valides dans la serie (n de 2 a 500):")
for p in [0, 1, 2, 3]:
    n_valid = 0
    for n in range(2, 502):
        val = log_iterated(mpf(n), p)
        if val > 0 and not isinf(val) and not isnan(val):
            n_valid += 1
    print(f"  p={p}: {n_valid} termes valides")


# =============================================================================
# TEST p=0 (REFERENCE CLASSIQUE)
# =============================================================================
print("\n" + "=" * 70)
print("TEST p=0 (Reference classique)")
print("=" * 70)

# Les zeros connus de zeta(s+1) = sum n^{-(s+1)}
# sont lies aux zeros de zeta classique decales
print("\nRecherche de zeros pour p=0...")

p0_results = []
for im_init in [5, 10, 14, 20, 25, 30]:
    result = find_zero_precise(0, mpc(0.5, im_init))
    p0_results.append(result)
    status = "CONV" if result['converged'] else "non-conv"
    print(f"  Init Im={im_init}: s = {result['real']:.6f} + {result['imag']:.2f}i [{status}]")


# =============================================================================
# TEST p=1
# =============================================================================
print("\n" + "=" * 70)
print("TEST p=1")
print("=" * 70)

print(f"Identite additive 0_1 = {float(additive_identity(1))}")

# Recherche sur grille
print("\nRecherche sur grille [0, 1.5] x [5, 40]...")
candidates_p1 = grid_search(1, (0, 1.5), (5, 40), re_step=0.1, im_step=2)
print(f"  {len(candidates_p1)} candidats trouves")

if candidates_p1:
    print("\nTop 5 candidats:")
    for c in candidates_p1[:5]:
        print(f"    s = {c['re']:.2f} + {c['im']:.1f}i, |zeta - 0_p| = {c['mag']:.4f}")

# Raffiner les meilleurs candidats
print("\nRaffinement des zeros...")
p1_results = []
refined_inits = [mpc(c['re'], c['im']) for c in candidates_p1[:8]]
# Ajouter des points standards
refined_inits.extend([mpc(0.5, 14), mpc(0.5, 21), mpc(0.5, 25)])

seen = set()
for s_init in refined_inits:
    key = (round(float(s_init.real), 1), round(float(s_init.imag), 0))
    if key in seen:
        continue
    seen.add(key)

    result = find_zero_precise(1, s_init)
    if result['converged'] or result['residual'] < 1e-5:
        # Verifier unicite
        is_new = True
        for prev in p1_results:
            if abs(prev['real'] - result['real']) < 0.05 and abs(prev['imag'] - result['imag']) < 0.5:
                is_new = False
                break
        if is_new:
            p1_results.append(result)
            dist = abs(result['real'] - 0.5)
            print(f"  Zero: s = {result['real']:.6f} + {result['imag']:.2f}i [dist a 0.5: {dist:.4f}]")


# =============================================================================
# TEST p=2
# =============================================================================
print("\n" + "=" * 70)
print("TEST p=2")
print("=" * 70)

print(f"Identite additive 0_2 = {float(additive_identity(2)):.6f} (= e)")

# Recherche sur grille
print("\nRecherche sur grille [0, 1.5] x [5, 50]...")
candidates_p2 = grid_search(2, (0, 1.5), (5, 50), re_step=0.1, im_step=2)
print(f"  {len(candidates_p2)} candidats trouves")

if candidates_p2:
    print("\nTop 5 candidats:")
    for c in candidates_p2[:5]:
        print(f"    s = {c['re']:.2f} + {c['im']:.1f}i, |zeta - 0_p| = {c['mag']:.4f}")

# Raffiner
print("\nRaffinement des zeros...")
p2_results = []
refined_inits = [mpc(c['re'], c['im']) for c in candidates_p2[:10]]
refined_inits.extend([mpc(0.5, 14), mpc(0.5, 21), mpc(0.5, 30), mpc(0.3, 15), mpc(0.7, 15)])

seen = set()
for s_init in refined_inits:
    key = (round(float(s_init.real), 1), round(float(s_init.imag), 0))
    if key in seen:
        continue
    seen.add(key)

    result = find_zero_precise(2, s_init)
    if result['converged'] or result['residual'] < 1e-5:
        is_new = True
        for prev in p2_results:
            if abs(prev['real'] - result['real']) < 0.05 and abs(prev['imag'] - result['imag']) < 0.5:
                is_new = False
                break
        if is_new:
            p2_results.append(result)
            dist = abs(result['real'] - 0.5)
            print(f"  Zero: s = {result['real']:.6f} + {result['imag']:.2f}i [dist a 0.5: {dist:.4f}]")


# =============================================================================
# TEST p=3
# =============================================================================
print("\n" + "=" * 70)
print("TEST p=3")
print("=" * 70)

print(f"Identite additive 0_3 = {float(additive_identity(3)):.4f}")

# Pour p=3, log^3(n) = log(log(log(n))) n'est valide que pour n > e^e ~ 15.15
print(f"Note: log^3(n) valide seulement pour n > e^e = {float(exp(exp(1))):.2f}")

# Recherche sur grille
print("\nRecherche sur grille [0, 1.5] x [5, 60]...")
candidates_p3 = grid_search(3, (0, 1.5), (5, 60), re_step=0.1, im_step=3)
print(f"  {len(candidates_p3)} candidats trouves")

if candidates_p3:
    print("\nTop 5 candidats:")
    for c in candidates_p3[:5]:
        print(f"    s = {c['re']:.2f} + {c['im']:.1f}i, |zeta - 0_p| = {c['mag']:.4f}")

# Raffiner
print("\nRaffinement des zeros...")
p3_results = []
refined_inits = [mpc(c['re'], c['im']) for c in candidates_p3[:10]]
refined_inits.extend([mpc(0.5, 14), mpc(0.5, 25), mpc(0.5, 35), mpc(0.3, 20), mpc(0.7, 20)])

seen = set()
for s_init in refined_inits:
    key = (round(float(s_init.real), 1), round(float(s_init.imag), 0))
    if key in seen:
        continue
    seen.add(key)

    result = find_zero_precise(3, s_init)
    if result['converged'] or result['residual'] < 1e-5:
        is_new = True
        for prev in p3_results:
            if abs(prev['real'] - result['real']) < 0.05 and abs(prev['imag'] - result['imag']) < 0.5:
                is_new = False
                break
        if is_new:
            p3_results.append(result)
            dist = abs(result['real'] - 0.5)
            print(f"  Zero: s = {result['real']:.6f} + {result['imag']:.2f}i [dist a 0.5: {dist:.4f}]")


# =============================================================================
# SYNTHESE
# =============================================================================
print("\n" + "=" * 70)
print("SYNTHESE COMPARATIVE")
print("=" * 70)

all_results = {
    'p0': p0_results,
    'p1': p1_results,
    'p2': p2_results,
    'p3': p3_results
}

print(f"\n{'Niveau':<8} {'Zeros':<8} {'Re(s) moyen':<15} {'Ecart-type':<12} {'Dist moy a 0.5':<15}")
print("-" * 60)

summary_stats = {}

for p, results in [(0, p0_results), (1, p1_results), (2, p2_results), (3, p3_results)]:
    valid = [r for r in results if r['converged'] or r['residual'] < 1e-5]

    if valid:
        re_vals = [r['real'] for r in valid]
        mean_re = np.mean(re_vals)
        std_re = np.std(re_vals)
        mean_dist = np.mean([abs(r - 0.5) for r in re_vals])

        summary_stats[p] = {
            'n_zeros': len(valid),
            'mean_re': mean_re,
            'std_re': std_re,
            'mean_dist': mean_dist,
            'zeros': valid
        }

        print(f"p={p:<6} {len(valid):<8} {mean_re:<15.6f} {std_re:<12.6f} {mean_dist:<15.6f}")
    else:
        summary_stats[p] = {'n_zeros': 0}
        print(f"p={p:<6} 0        N/A")


# =============================================================================
# VERIFICATION HYPOTHESE RIEMANN
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION HYPOTHESE DE RIEMANN GENERALISEE")
print("=" * 70)

print("\nDistribution des Re(s) par niveau:")
for p in [0, 1, 2, 3]:
    if summary_stats[p]['n_zeros'] > 0:
        zeros = summary_stats[p]['zeros']
        on_critical = sum(1 for z in zeros if abs(z['real'] - 0.5) < 0.05)
        near_critical = sum(1 for z in zeros if abs(z['real'] - 0.5) < 0.15)

        print(f"\n  p={p}:")
        print(f"    Total zeros: {len(zeros)}")
        print(f"    Sur ligne critique (|Re-0.5| < 0.05): {on_critical} ({100*on_critical/len(zeros):.0f}%)")
        print(f"    Pres de la ligne (|Re-0.5| < 0.15): {near_critical} ({100*near_critical/len(zeros):.0f}%)")

        if zeros:
            print(f"    Re(s) min: {min(z['real'] for z in zeros):.4f}")
            print(f"    Re(s) max: {max(z['real'] for z in zeros):.4f}")


# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Analyser la tendance
if all(summary_stats[p]['n_zeros'] > 0 for p in [1, 2, 3]):
    means = [summary_stats[p]['mean_re'] for p in [1, 2, 3]]
    dists = [summary_stats[p]['mean_dist'] for p in [1, 2, 3]]

    print(f"\nEvolution de Re(s) moyen: p=1: {means[0]:.4f}, p=2: {means[1]:.4f}, p=3: {means[2]:.4f}")
    print(f"Evolution distance a 0.5: p=1: {dists[0]:.4f}, p=2: {dists[1]:.4f}, p=3: {dists[2]:.4f}")

    if dists[2] < dists[1] < dists[0]:
        print("\n>>> TENDANCE: Les zeros se RAPPROCHENT de la ligne critique quand p augmente")
        print(">>> Cela suggere une convergence vers Re(s) = 0.5 pour p -> infini")
    elif dists[2] > dists[1] > dists[0]:
        print("\n>>> TENDANCE: Les zeros s'ELOIGNENT de la ligne critique quand p augmente")
    else:
        print("\n>>> TENDANCE: Pas de pattern clair")

# Sauvegarder
output_data = {
    'summary': {str(p): {k: v for k, v in s.items() if k != 'zeros'}
                for p, s in summary_stats.items()},
    'zeros': {str(p): s.get('zeros', []) for p, s in summary_stats.items()},
    'timestamp': datetime.now().isoformat()
}

with open(OUTPUT_DIR / "p2_p3_deep_results.json", 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

print(f"\nResultats sauvegardes: {OUTPUT_DIR / 'p2_p3_deep_results.json'}")

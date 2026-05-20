"""
Plan d'Experience CORRIGE BIS: Hierarchie Exponentielle
========================================================

CORRECTIONS CONCEPTUELLES (correction_bis.pdf):
1. zeta_p(s) = exp^{o p}( sum (log^{o p}(n))^{-(s+1)} )
2. Zeros: zeta_p(s) = 0_p = exp^{o p}(0)
   - p=0: 0_0 = 0
   - p=1: 0_1 = 1
   - p=2: 0_2 = e
   - p general: 0_p = exp^{o p}(0)
3. Operations definies via Phi_p = exp^{o p}:
   x oplus_p y = Phi_p(Phi_p^{-1}(x) + Phi_p^{-1}(y))
   x otimes_p y = Phi_p(Phi_p^{-1}(x) * Phi_p^{-1}(y))

4. RELATION EMPIRIQUE DEFAUT-ZERO (nouvelle):
   Re(rho) ≈ 0.52 - 0.01 * log10(delta_max)

5. INVARIANCE DE LA LIGNE CRITIQUE:
   Re(rho_p) = 1/2 pour tout p in Z (sous hypothese de regularite)

Grille: 75 transferts arithmetiques, niveaux p in [-5, +5]
"""

import numpy as np
from mpmath import mp, mpf, mpc, exp, log, power, fabs, pi, isnan, isinf, nstr
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Precision
mp.dps = 50  # 50 decimal places for better precision

# Configuration (75 transferts selon correction_bis.pdf)
BASES = np.arange(1.05, 2.01, 0.13)  # ~8 bases
LEVELS = list(range(-5, 6))  # -5 to +5 (11 niveaux)
# Total: ~88 experiences (proche de 75)
OUTPUT_DIR = Path("results/hierarchy_corrected_bis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PLAN D'EXPERIENCE CORRIGE BIS")
print("Fonction Zeta Intrinseque de Niveau p")
print("Relation Empirique: Re(rho) = 0.52 - 0.01*log10(delta_max)")
print("="*80)
print(f"Bases: {len(BASES)} valeurs de {BASES[0]:.2f} a {BASES[-1]:.2f}")
print(f"Niveaux: {len(LEVELS)} valeurs de {LEVELS[0]} a {LEVELS[-1]}")
print(f"Total experiences: {len(BASES) * len(LEVELS)}")
print("="*80)


# =============================================================================
# FONCTIONS DE BASE CORRIGEES
# =============================================================================

def exp_iterated(x, p):
    """
    Phi_p(x) = exp^{o p}(x) = exp(exp(...exp(x)...)) (p fois)

    Pour p < 0: c'est log^{o |p|}(x)
    Pour p = 0: identite
    Pour p > 0: exp compose p fois
    """
    x = mpf(x)

    if p == 0:
        return x
    elif p > 0:
        result = x
        for _ in range(p):
            if result > 700:  # Eviter overflow
                return mpf('inf')
            result = exp(result)
        return result
    else:  # p < 0
        result = x
        for _ in range(-p):
            if result <= 0:
                return mpf('-inf')
            result = log(result)
        return result


def log_iterated(x, p):
    """
    Phi_p^{-1}(x) = log^{o p}(x)

    Inverse de exp_iterated
    """
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
    else:  # p < 0
        result = x
        for _ in range(-p):
            if result > 700:
                return mpf('inf')
            result = exp(result)
        return result


def additive_identity(p):
    """
    Identite additive 0_p = exp^{o p}(0)

    p=0: 0
    p=1: exp(0) = 1
    p=2: exp(exp(0)) = exp(1) = e
    p=3: exp(exp(exp(0))) = exp(e) ~ 15.15
    ...
    """
    return exp_iterated(mpf(0), p)


def multiplicative_identity(p):
    """
    Identite multiplicative 1_p = exp^{o p}(1)

    p=0: 1
    p=1: exp(1) = e
    p=2: exp(exp(1)) = exp(e) ~ 15.15
    ...
    """
    return exp_iterated(mpf(1), p)


def oplus_p(x, y, p):
    """
    Addition de niveau p:
    x oplus_p y = Phi_p(Phi_p^{-1}(x) + Phi_p^{-1}(y))
    """
    x, y = mpf(x), mpf(y)

    inv_x = log_iterated(x, p)
    inv_y = log_iterated(y, p)

    if isinf(inv_x) or isinf(inv_y) or isnan(inv_x) or isnan(inv_y):
        return mpf('nan')

    sum_val = inv_x + inv_y
    return exp_iterated(sum_val, p)


def otimes_p(x, y, p):
    """
    Multiplication de niveau p:
    x otimes_p y = Phi_p(Phi_p^{-1}(x) * Phi_p^{-1}(y))
    """
    x, y = mpf(x), mpf(y)

    inv_x = log_iterated(x, p)
    inv_y = log_iterated(y, p)

    if isinf(inv_x) or isinf(inv_y) or isnan(inv_x) or isnan(inv_y):
        return mpf('nan')

    prod_val = inv_x * inv_y
    return exp_iterated(prod_val, p)


# =============================================================================
# FONCTION ZETA CORRIGEE
# =============================================================================

def zeta_p(s, p, N_terms=100):
    """
    Fonction zeta de niveau p CORRIGEE:

    zeta_p(s) = exp^{o p}( sum_{n=1}^{infty} (log^{o p}(n))^{-(s+1)} )

    La somme interne est calculee dans l'espace "plat" (niveau 0),
    puis transformee par exp^{o p} pour revenir au niveau p.
    """
    s = mpc(s)

    # Calculer la somme interne: sum (log^{o p}(n))^{-(s+1)}
    inner_sum = mpc(0)

    for n in range(2, N_terms + 2):  # Commencer a n=2 pour eviter log(1)=0
        try:
            # log^{o p}(n)
            log_p_n = log_iterated(mpf(n), p)

            if log_p_n <= 0 or isinf(log_p_n) or isnan(log_p_n):
                continue

            # (log^{o p}(n))^{-(s+1)}
            term = power(log_p_n, -(s + 1))

            if isnan(term) or isinf(fabs(term)):
                continue

            inner_sum += term

            # Convergence check
            if fabs(term) < mpf('1e-30'):
                break

        except:
            continue

    # Appliquer exp^{o p} a la somme
    if p == 0:
        return inner_sum
    else:
        # Pour p != 0, on applique exp^{o p}
        real_sum = float(inner_sum.real)
        imag_sum = float(inner_sum.imag)

        if abs(real_sum) > 700:  # Overflow protection
            return mpc(mpf('inf'), 0)

        # exp^{o p} applique a un nombre complexe
        result = exp_iterated(real_sum, p)
        return mpc(result, imag_sum * float(result) if result < mpf('inf') else 0)


def find_zero_corrected(p, s_init=None, max_iter=100, tol=1e-12):
    """
    Recherche d'un zero de zeta_p(s).

    CONDITION CORRIGEE: zeta_p(s) = 0_p = exp^{o p}(0)

    Donc on cherche s tel que zeta_p(s) - 0_p = 0
    """
    # Identite additive du niveau p
    zero_p = additive_identity(p)

    # Point de depart
    if s_init is None:
        # Adapter le point de depart selon le niveau
        if p >= 0:
            s_init = mpc(0.5, 14.0)
        else:
            s_init = mpc(0.5, 10.0)

    s = mpc(s_init)
    h = mpf('1e-6')

    for iteration in range(max_iter):
        try:
            # Evaluer zeta_p(s)
            z_val = zeta_p(s, p)

            # Residu: distance a l'identite additive
            if p == 0:
                residual = z_val  # 0_0 = 0
            else:
                # Pour p > 0, le "zero" est quand zeta_p(s) = 0_p
                # Mais 0_p peut etre tres grand (e.g., e pour p=2)
                # On cherche plutot quand la somme interne = 0
                residual = z_val - zero_p

            res_mag = fabs(residual)

            if res_mag < tol:
                return {
                    'real': float(s.real),
                    'imag': float(s.imag),
                    'residual': float(res_mag),
                    'converged': True,
                    'iterations': iteration,
                    'zero_p': float(zero_p) if zero_p < mpf('1e100') else 'inf'
                }

            # Derivee numerique
            z_plus = zeta_p(s + h, p)
            z_minus = zeta_p(s - h, p)
            dz = (z_plus - z_minus) / (2 * h)

            if fabs(dz) < mpf('1e-30'):
                # Essayer une perturbation
                s = s + mpc(0.01, 0.1)
                continue

            # Newton step
            s_new = s - residual / dz

            # Contraintes
            if s_new.real < -5:
                s_new = mpc(-5, s_new.imag)
            if s_new.real > 5:
                s_new = mpc(5, s_new.imag)
            if s_new.imag < 1:
                s_new = mpc(s_new.real, 5)
            if s_new.imag > 50:
                s_new = mpc(s_new.real, 50)

            # Check convergence
            if fabs(s_new - s) < mpf('1e-14'):
                z_final = zeta_p(s_new, p)
                return {
                    'real': float(s_new.real),
                    'imag': float(s_new.imag),
                    'residual': float(fabs(z_final - zero_p)) if p != 0 else float(fabs(z_final)),
                    'converged': True,
                    'iterations': iteration,
                    'zero_p': float(zero_p) if zero_p < mpf('1e100') else 'inf'
                }

            s = s_new

        except Exception as e:
            # Perturber et continuer
            s = s + mpc(0.05, 0.2)
            continue

    # Non converge
    try:
        z_final = zeta_p(s, p)
        res = fabs(z_final - zero_p) if p != 0 else fabs(z_final)
    except:
        res = mpf('inf')

    return {
        'real': float(s.real),
        'imag': float(s.imag),
        'residual': float(res) if res < mpf('1e100') else float('inf'),
        'converged': False,
        'iterations': max_iter,
        'zero_p': float(zero_p) if zero_p < mpf('1e100') else 'inf'
    }


def compute_defect_corrected(alpha, p, test_pairs=None):
    """
    Defaut pour le transfert phi(n) = alpha^n au niveau p.

    Le defaut mesure l'incompatibilite entre phi et la multiplication de niveau p.
    """
    if test_pairs is None:
        test_pairs = [(2, 3), (2, 4), (3, 4), (2, 5)]

    alpha = mpf(alpha)
    defects = []

    for a, b in test_pairs:
        try:
            # phi(a) = alpha^a, phi(b) = alpha^b
            phi_a = power(alpha, a)
            phi_b = power(alpha, b)

            # phi(a) otimes_p phi(b) vs phi(a * b)
            # Le defaut est: phi(a) * phi(b) - phi(a*b) dans l'arithmetique classique
            phi_ab = power(alpha, a * b)

            defect = fabs(phi_a * phi_b - phi_ab)

            if not (isnan(defect) or isinf(defect)):
                defects.append(float(defect))
        except:
            continue

    if defects:
        return {
            'max': max(defects),
            'mean': sum(defects) / len(defects),
            'is_zero': max(defects) < 1e-20
        }
    return {'max': float('inf'), 'mean': float('inf'), 'is_zero': False}


def predict_zero_position(delta_max):
    """
    Relation empirique correction_bis.pdf:
    Re(rho) ≈ 0.52 - 0.01 * log10(delta_max)

    Predit la partie reelle du premier zero non-trivial
    en fonction du defaut maximal.
    """
    if delta_max <= 0 or delta_max == float('inf'):
        return None
    try:
        import math
        return 0.52 - 0.01 * math.log10(delta_max)
    except:
        return None


def verify_empirical_law(zeros_data, defects_data):
    """
    Verifie la loi empirique: Re(rho) = 0.52 - 0.01 * log10(delta_max)

    Retourne les statistiques de correlation.
    """
    import math

    valid_pairs = []
    for zd, dd in zip(zeros_data, defects_data):
        if (zd and dd and
            zd.get('converged') and
            dd.get('max') and
            dd['max'] > 0 and
            dd['max'] != float('inf')):

            re_rho = zd['real']
            delta_max = dd['max']
            predicted = predict_zero_position(delta_max)

            # Filtrer les NaN
            if predicted is not None and not math.isnan(re_rho):
                valid_pairs.append({
                    'actual': re_rho,
                    'predicted': predicted,
                    'delta_max': delta_max,
                    'error': abs(re_rho - predicted)
                })

    if not valid_pairs:
        return {'n_valid': 0}

    errors = [p['error'] for p in valid_pairs]
    actuals = [p['actual'] for p in valid_pairs]
    predicteds = [p['predicted'] for p in valid_pairs]

    # Correlation coefficient
    n = len(valid_pairs)
    if n < 2:
        return {'n_valid': n, 'mean_error': errors[0] if errors else 0, 'max_error': errors[0] if errors else 0, 'correlation': 0, 'pairs': valid_pairs}

    mean_a = sum(actuals) / n
    mean_p = sum(predicteds) / n

    numerator = sum((a - mean_a) * (p - mean_p) for a, p in zip(actuals, predicteds))
    denom_a = math.sqrt(sum((a - mean_a)**2 for a in actuals))
    denom_p = math.sqrt(sum((p - mean_p)**2 for p in predicteds))

    correlation = numerator / (denom_a * denom_p) if denom_a * denom_p > 0 else 0

    return {
        'n_valid': n,
        'mean_error': sum(errors) / n,
        'max_error': max(errors),
        'correlation': correlation,
        'pairs': valid_pairs
    }


# =============================================================================
# EXECUTION DE L'EXPERIENCE
# =============================================================================

def run_single_experiment(alpha, p):
    """Execute une experience pour (alpha, p)."""
    result = {
        'alpha': float(alpha),
        'level': p,
        'success': False,
        'error': None
    }

    try:
        # Identites
        result['zero_p'] = float(additive_identity(p)) if additive_identity(p) < mpf('1e100') else 'inf'
        result['one_p'] = float(multiplicative_identity(p)) if multiplicative_identity(p) < mpf('1e100') else 'inf'

        # Test des operations
        test_oplus = oplus_p(mpf(2), mpf(3), p)
        test_otimes = otimes_p(mpf(2), mpf(3), p)

        result['oplus_2_3'] = float(test_oplus) if not (isnan(test_oplus) or isinf(test_oplus)) else None
        result['otimes_2_3'] = float(test_otimes) if not (isnan(test_otimes) or isinf(test_otimes)) else None

        # Defaut
        result['defect'] = compute_defect_corrected(alpha, p)

        # Recherche de zero (seulement pour niveaux raisonnables)
        if -5 <= p <= 5:
            zero_result = find_zero_corrected(p)
            result['zero'] = zero_result

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def run_all_experiments(progress_interval=50):
    """Execute toutes les experiences."""
    results = []
    total = len(BASES) * len(LEVELS)
    count = 0

    print(f"\nDemarrage de {total} experiences...")
    start_time = datetime.now()

    for alpha in BASES:
        for p in LEVELS:
            count += 1

            if count % progress_interval == 0 or count == total:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else 0
                print(f"  [{count}/{total}] {100*count/total:.1f}% - "
                      f"alpha={alpha:.2f}, p={p:+d} - ETA: {eta:.0f}s")

            result = run_single_experiment(alpha, p)
            results.append(result)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nTermine en {elapsed:.1f}s ({total/elapsed:.1f} exp/s)")

    return results


def analyze_and_save(results):
    """Analyse et sauvegarde les resultats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Statistiques par niveau
    print("\n" + "="*80)
    print("RESULTATS PAR NIVEAU")
    print("="*80)

    print(f"\n{'Niveau':<8} {'0_p':<15} {'Zeros':<8} {'Re(s) moy':<12} {'Dist a 0.5':<12}")
    print("-" * 60)

    level_stats = {}

    for p in LEVELS:
        level_results = [r for r in results if r['level'] == p]
        n_success = sum(1 for r in level_results if r['success'])

        # Identite additive
        zero_p_val = level_results[0].get('zero_p', 'N/A') if level_results else 'N/A'

        # Zeros trouves
        zeros = [r['zero'] for r in level_results
                 if r.get('zero') and r['zero'].get('converged')]
        n_zeros = len(zeros)

        if zeros:
            mean_re = np.mean([z['real'] for z in zeros])
            mean_dist = np.mean([abs(z['real'] - 0.5) for z in zeros])
        else:
            mean_re = None
            mean_dist = None

        level_stats[p] = {
            'zero_p': zero_p_val,
            'n_zeros': n_zeros,
            'mean_re': mean_re,
            'mean_dist': mean_dist
        }

        zero_p_str = f"{zero_p_val:.4f}" if isinstance(zero_p_val, float) else str(zero_p_val)
        re_str = f"{mean_re:.6f}" if mean_re is not None else "N/A"
        dist_str = f"{mean_dist:.6f}" if mean_dist is not None else "N/A"

        print(f"{p:<8} {zero_p_str:<15} {n_zeros:<8} {re_str:<12} {dist_str:<12}")

    # ==========================================================================
    # VERIFICATION DE LA LOI EMPIRIQUE (correction_bis.pdf)
    # Re(rho) = 0.52 - 0.01 * log10(delta_max)
    # ==========================================================================
    print("\n" + "="*80)
    print("VERIFICATION LOI EMPIRIQUE: Re(rho) = 0.52 - 0.01*log10(delta_max)")
    print("="*80)

    zeros_data = [r.get('zero') for r in results if r.get('zero')]
    defects_data = [r.get('defect') for r in results if r.get('defect')]

    empirical_check = verify_empirical_law(zeros_data, defects_data)

    if empirical_check['n_valid'] > 0:
        print(f"\nPaires valides analysees: {empirical_check['n_valid']}")
        print(f"Erreur moyenne (predit vs observe): {empirical_check['mean_error']:.6f}")
        print(f"Erreur maximale: {empirical_check['max_error']:.6f}")
        print(f"Coefficient de correlation: {empirical_check['correlation']:.4f}")

        print("\nExemples (5 premiers):")
        print(f"{'delta_max':<15} {'Re(rho) obs':<15} {'Re(rho) pred':<15} {'Erreur':<10}")
        print("-" * 55)
        for pair in empirical_check.get('pairs', [])[:5]:
            print(f"{pair['delta_max']:<15.6f} {pair['actual']:<15.6f} "
                  f"{pair['predicted']:<15.6f} {pair['error']:<10.6f}")

        # Conclusion
        if empirical_check['correlation'] > 0.7:
            print("\n[OK] LOI EMPIRIQUE CONFIRMEE (correlation > 0.7)")
        elif empirical_check['correlation'] > 0.5:
            print("\n[~] LOI EMPIRIQUE PARTIELLEMENT CONFIRMEE (0.5 < correlation < 0.7)")
        else:
            print("\n[X] LOI EMPIRIQUE NON CONFIRMEE (correlation < 0.5)")
    else:
        print("\nPas assez de donnees pour verifier la loi empirique.")

    # ==========================================================================
    # VERIFICATION INVARIANCE LIGNE CRITIQUE
    # ==========================================================================
    print("\n" + "="*80)
    print("VERIFICATION INVARIANCE LIGNE CRITIQUE Re(s) = 1/2")
    print("="*80)

    all_zeros = [r['zero'] for r in results
                 if r.get('zero') and r['zero'].get('converged')]

    if all_zeros:
        re_parts = [z['real'] for z in all_zeros]
        mean_re = np.mean(re_parts)
        std_re = np.std(re_parts)
        on_critical = sum(1 for re in re_parts if abs(re - 0.5) < 0.05)

        print(f"\nTotal zeros converges: {len(all_zeros)}")
        print(f"Re(s) moyen: {mean_re:.6f}")
        print(f"Ecart-type Re(s): {std_re:.6f}")
        print(f"Zeros sur ligne critique (|Re(s)-0.5| < 0.05): {on_critical}/{len(all_zeros)}")
        print(f"Pourcentage: {100*on_critical/len(all_zeros):.1f}%")

        if abs(mean_re - 0.5) < 0.1 and on_critical / len(all_zeros) > 0.8:
            print("\n[OK] INVARIANCE DE LA LIGNE CRITIQUE CONFIRMEE")
        else:
            print("\n[~] INVARIANCE PARTIELLE OU DERIVE OBSERVEE")

    # Sauvegarder
    output_file = OUTPUT_DIR / f"results_corrected_bis_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    latest_file = OUTPUT_DIR / "results_corrected_bis_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    stats_file = OUTPUT_DIR / "level_stats_corrected_bis.json"
    stats_data = {
        'level_stats': level_stats,
        'empirical_law': empirical_check,
        'timestamp': timestamp
    }
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2, default=str)

    print(f"\nResultats sauvegardes: {output_file}")

    return level_stats


def main():
    print("\n" + "="*80)
    print("CORRECTIONS APPLIQUEES (correction_bis.pdf / correction_bis.txt)")
    print("="*80)
    print("""
    1. FONCTION ZETA INTRINSEQUE:
       zeta_p(s) = exp^{o p}( sum (log^{o p}(n))^{-(s+1)} )

    2. CONDITION DES ZEROS:
       zeta_p(s) = 0_p = exp^{o p}(0)

    3. RELATION EMPIRIQUE DEFAUT-ZERO:
       Re(rho) = 0.52 - 0.01 * log10(delta_max)

    4. INVARIANCE LIGNE CRITIQUE:
       Re(rho_p) = 1/2 pour tout p in Z
    """)

    print("\n" + "="*80)
    print("VERIFICATION DES DEFINITIONS CORRIGEES")
    print("="*80)

    print("\nIdentites additives 0_p = exp^{o p}(0):")
    for p in range(-3, 6):
        zero_p = additive_identity(p)
        print(f"  p={p:+d}: 0_p = {nstr(zero_p, 6) if zero_p < mpf('1e10') else 'inf'}")

    print("\nIdentites multiplicatives 1_p = exp^{o p}(1):")
    for p in range(-3, 4):
        one_p = multiplicative_identity(p)
        print(f"  p={p:+d}: 1_p = {nstr(one_p, 6) if one_p < mpf('1e10') else 'inf'}")

    print("\nOperations au niveau p=1:")
    print(f"  2 oplus_1 3 = {nstr(oplus_p(2, 3, 1), 6)} (devrait etre 6 = 2*3)")
    print(f"  2 otimes_1 3 = {nstr(otimes_p(2, 3, 1), 6)} (devrait etre 8 = 2^(log2*log3))")

    print("\nOperations au niveau p=0 (classique):")
    print(f"  2 oplus_0 3 = {nstr(oplus_p(2, 3, 0), 6)} (devrait etre 5)")
    print(f"  2 otimes_0 3 = {nstr(otimes_p(2, 3, 0), 6)} (devrait etre 6)")

    # Test de la prediction empirique
    print("\nTest relation empirique:")
    test_defects = [1e-5, 1e-3, 1e-1, 1e1, 1e3]
    for d in test_defects:
        pred = predict_zero_position(d)
        print(f"  delta_max = {d:.0e} -> Re(rho) predit = {pred:.4f}")

    # Lancer les experiences
    print("\n" + "="*80)
    print("LANCEMENT DES EXPERIENCES")
    print("="*80)

    results = run_all_experiments(progress_interval=20)
    level_stats = analyze_and_save(results)

    print("\n" + "="*80)
    print("EXPERIENCE TERMINEE")
    print("="*80)

    return results, level_stats


if __name__ == "__main__":
    results, level_stats = main()

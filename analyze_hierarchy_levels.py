"""
Analyse des niveaux consecutifs de la hierarchie exponentielle
Focus sur les paires (-2,-1), (-1,0), (0,1), (1,2)
"""

import json
import numpy as np

with open('results/full_experiment/results_latest.json', 'r') as f:
    results = json.load(f)

# Filtrer les resultats de la hierarchie
hierarchy = [r for r in results if r['category'] == 'Hierarchy']

print("="*80)
print("ANALYSE DES NIVEAUX CONSECUTIFS - HIERARCHIE EXPONENTIELLE")
print("="*80)

print("""
RAPPEL THEORIQUE:
================
- L'operation oplus_n est definie par: x oplus_n y = exp^(n)(log^(n)(x) + log^(n)(y))
- oplus_0 = addition classique (+)
- oplus_1 = multiplication classique (x)
- oplus_2 = exp(log(x) * log(y))

POINT CLE: Entre deux niveaux CONSECUTIFS (n, n+1), le defaut est NUL.
C'est-a-dire que (oplus_n, oplus_{n+1}) forme une arithmetique coherente.

La question: Comment se comportent les zeros pour chaque NIVEAU individuel?
""")

# Organiser par base et niveau
bases = ['basee', 'base2', 'base10']

for base in bases:
    print(f"\n{'='*80}")
    print(f"BASE: {base.upper()}")
    print("="*80)

    # Extraire les niveaux pour cette base
    base_results = {}
    for r in hierarchy:
        if base in r['name']:
            # Extraire le niveau
            for part in r['name'].split('_'):
                if 'level' in part:
                    level = int(part.replace('level', ''))
                    base_results[level] = r

    # Afficher par niveau
    print(f"\n{'Niveau':<10} {'Defaut Max':<15} {'Defaut=0?':<12} {'Zeros':<8} {'Re(s)':<12} {'Dist 0.5':<12} {'Cocycle'}")
    print("-" * 85)

    for level in sorted(base_results.keys()):
        r = base_results[level]
        defect_max = r['defect']['max']
        defect_zero = r['defect']['is_zero']
        n_zeros = r['n_zeros']

        if n_zeros > 0:
            re_s = r['zeros'][0]['real']
            dist = abs(re_s - 0.5)
            re_str = f"{re_s:.6f}"
            dist_str = f"{dist:.6f}"
        else:
            re_str = "N/A"
            dist_str = "N/A"

        cocycle = "Oui" if r['cohomology']['cocycle_verified'] else "Non"

        if defect_max == float('inf'):
            defect_str = "inf"
        else:
            defect_str = f"{defect_max:.2e}"

        defect_zero_str = "OUI" if defect_zero else "NON"

        print(f"{level:<10} {defect_str:<15} {defect_zero_str:<12} {n_zeros:<8} {re_str:<12} {dist_str:<12} {cocycle}")

print("\n" + "="*80)
print("ANALYSE DES PAIRES CONSECUTIVES")
print("="*80)

print("""
Pour chaque paire (n, n+1), la theorie predit un defaut NUL entre les operations.
Analysons ce que cela signifie pour les zeros:
""")

# Analyser les transitions
for base in bases:
    print(f"\n--- {base.upper()} ---")

    base_results = {}
    for r in hierarchy:
        if base in r['name']:
            for part in r['name'].split('_'):
                if 'level' in part:
                    level = int(part.replace('level', ''))
                    base_results[level] = r

    levels = sorted(base_results.keys())

    for i in range(len(levels) - 1):
        n = levels[i]
        n1 = levels[i + 1]

        r_n = base_results[n]
        r_n1 = base_results[n1]

        print(f"\nTransition ({n}, {n+1}):")
        print(f"  Niveau {n}: defaut={'0' if r_n['defect']['is_zero'] else 'non-zero'}, ", end="")
        if r_n['n_zeros'] > 0:
            print(f"Re(s)={r_n['zeros'][0]['real']:.4f}")
        else:
            print("pas de zero trouve")

        print(f"  Niveau {n+1}: defaut={'0' if r_n1['defect']['is_zero'] else 'non-zero'}, ", end="")
        if r_n1['n_zeros'] > 0:
            print(f"Re(s)={r_n1['zeros'][0]['real']:.4f}")
        else:
            print("pas de zero trouve")

        # Interpretation
        if n == 0:
            print(f"  >>> C'est l'ARITHMETIQUE CLASSIQUE (addition -> multiplication)")

        if r_n['n_zeros'] > 0 and r_n1['n_zeros'] > 0:
            delta_re = r_n1['zeros'][0]['real'] - r_n['zeros'][0]['real']
            print(f"  Delta Re(s) = {delta_re:+.4f}")

print("\n" + "="*80)
print("INTERPRETATION PHYSIQUE")
print("="*80)

print("""
OBSERVATIONS CLES:
==================

1. NIVEAU 0 (Arithmetique classique: + et x)
   - Defaut = 0 (par definition, c'est le cas de reference)
   - Les zeros trouves sont a Re(s) ~ 0.34, PAS a 0.5
   - Ceci suggere que la fonction zeta twistee zeta_alpha(s) = sum(alpha^{-ns}/n^s)
     est DIFFERENTE de la fonction zeta de Riemann classique

2. NIVEAUX NEGATIFS (-2, -1)
   - Ce sont les "sous-operations" (racines, logarithmes)
   - Niveau -1: zeros plus proches de 0.5 (Re ~ 0.51)
   - Defaut non-nul individuellement

3. NIVEAUX POSITIFS (1, 2)
   - Ce sont les "super-operations" (exponentiation, tour d'exponentielles)
   - Zeros s'eloignent de 0.5 avec le niveau
   - Defaut explose (tend vers l'infini)

4. POINT CRUCIAL - LA PAIRE (0,1):
   - C'est l'arithmetique classique (addition, multiplication)
   - Entre niveau 0 et niveau 1, le defaut DEVRAIT etre zero
   - MAIS individuellement:
     * Niveau 0: Re(s) ~ 0.34
     * Niveau 1: Re(s) varie selon la base (0.22 a 0.36)
   - Les zeros NE SONT PAS sur la ligne critique Re=0.5
""")

# Focus sur le cas classique
print("\n" + "="*80)
print("FOCUS: LE CAS CLASSIQUE (0,1)")
print("="*80)

for base in bases:
    base_results = {}
    for r in hierarchy:
        if base in r['name']:
            for part in r['name'].split('_'):
                if 'level' in part:
                    level = int(part.replace('level', ''))
                    base_results[level] = r

    if 0 in base_results and 1 in base_results:
        r0 = base_results[0]
        r1 = base_results[1]

        print(f"\n{base.upper()}:")
        print(f"  Niveau 0 (addition): Re(s) = {r0['zeros'][0]['real']:.6f}" if r0['n_zeros'] > 0 else "  Niveau 0: pas de zero")
        print(f"  Niveau 1 (multiplication): Re(s) = {r1['zeros'][0]['real']:.6f}" if r1['n_zeros'] > 0 else "  Niveau 1: pas de zero")

        if r0['n_zeros'] > 0 and r1['n_zeros'] > 0:
            print(f"  Distance entre les zeros: {abs(r1['zeros'][0]['real'] - r0['zeros'][0]['real']):.6f}")
            print(f"  Distance moyenne a 0.5: {(abs(r0['zeros'][0]['real']-0.5) + abs(r1['zeros'][0]['real']-0.5))/2:.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
La difference de comportement entre les niveaux consecutifs revele:

1. Le DEFAUT NUL entre niveaux consecutifs ne garantit PAS que les zeros
   sont sur la ligne critique Re=0.5.

2. Chaque niveau definit une fonction zeta DIFFERENTE avec ses propres zeros.

3. La transition (0,1) - l'arithmetique classique - montre que:
   - Le niveau 0 (addition seule) donne des zeros a Re ~ 0.34
   - Le niveau 1 (multiplication) donne des zeros qui varient selon alpha

4. IMPLICATION POUR RH:
   La ligne critique Re=0.5 semble etre une propriete SPECIFIQUE de la
   fonction zeta de Riemann classique zeta(s) = sum(1/n^s), et NON une
   propriete universelle des fonctions zeta twistees zeta_alpha(s).

5. La coherence cohomologique (defaut=0 entre niveaux) est une propriete
   ALGEBRIQUE qui ne determine pas directement la position ANALYTIQUE
   des zeros.
""")

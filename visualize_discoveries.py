"""
Visualisation des decouvertes sur la fonction zeta modifiee
===========================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'

OUTPUT_DIR = Path("results/discoveries_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("VISUALISATION DES DECOUVERTES")
print("=" * 70)

# Donnees des tests
zeros_by_level = {
    0: [(0.389, 23.31)],
    1: [(0.959, 14.55), (1.155, 20.81), (0.911, 23.57), (0.616, 30.34), (0.578, 35.94)],
    2: [(0.565, 14.51), (0.501, 21.22), (0.431, 25.56), (0.366, 29.85), (0.294, 34.19)],
    3: [(0.328, 14.60), (0.376, 20.60), (0.087, 24.39), (0.400, 30.26)],
    4: [(0.500, 14.0), (0.500, 21.0), (0.500, 25.0)]  # Artefact
}

# =============================================================================
# Figure 1: Position des zeros dans le plan complexe
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 8))

colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd'}
markers = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'x'}

for p, zeros in zeros_by_level.items():
    if zeros:
        re_vals = [z[0] for z in zeros]
        im_vals = [z[1] for z in zeros]
        label = f'p={p}' if p < 4 else f'p={p} (artefact)'
        ax1.scatter(re_vals, im_vals, c=colors[p], marker=markers[p],
                   s=120, label=label, alpha=0.8, edgecolors='black', linewidth=0.5)

# Ligne critique
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Re(s) = 1/2')

ax1.set_xlabel('Re(s)', fontsize=14)
ax1.set_ylabel('Im(s)', fontsize=14)
ax1.set_title('Zeros de $D_p(s) = \sum (\log^{\\circ p}(n))^{-(s+1)}$\npar niveau arithmetique', fontsize=14)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.1, 1.6)

plt.tight_layout()
fig1.savefig(OUTPUT_DIR / "01_zeros_all_levels.png", dpi=150)
print("[1/6] zeros_all_levels.png")
plt.close()

# =============================================================================
# Figure 2: Re(s) moyen vs niveau p
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

levels = [0, 1, 2, 3]
mean_re = []
std_re = []

for p in levels:
    re_vals = [z[0] for z in zeros_by_level[p]]
    mean_re.append(np.mean(re_vals))
    std_re.append(np.std(re_vals))

bars = ax2.bar(levels, mean_re, yerr=std_re, capsize=5,
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
               edgecolor='black', linewidth=1.5, alpha=0.8)

ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Re(s) = 1/2')

# Annotations
for i, (p, m) in enumerate(zip(levels, mean_re)):
    ax2.annotate(f'{m:.2f}', (p, m + std_re[i] + 0.05), ha='center', fontsize=12, fontweight='bold')

ax2.set_xlabel('Niveau p', fontsize=14)
ax2.set_ylabel('Re(s) moyen', fontsize=14)
ax2.set_title('Evolution de Re(s) moyen selon le niveau arithmetique\n(p=2 est le plus proche de la ligne critique)', fontsize=14)
ax2.set_xticks(levels)
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.2)

plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "02_re_vs_level.png", dpi=150)
print("[2/6] re_vs_level.png")
plt.close()

# =============================================================================
# Figure 3: Cas p=4 - Demonstration de l'artefact
# =============================================================================
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3a: Serie interne par niveau
ax3a = axes[0]
p_vals = [0, 1, 2, 3, 4, 5]
inner_sum_mags = [0.448, 5.479, 55.98, 226.2, 0.0, 0.0]

colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = ax3a.bar(p_vals, inner_sum_mags, color=colors_bar, edgecolor='black', linewidth=1)

# Marquer p>=4 comme artefact
for i in [4, 5]:
    bars[i].set_hatch('///')
    bars[i].set_alpha(0.5)

ax3a.set_xlabel('Niveau p', fontsize=12)
ax3a.set_ylabel('|Somme interne|', fontsize=12)
ax3a.set_title('Magnitude de la somme interne\n(p>=4: serie VIDE = artefact)', fontsize=12)
ax3a.set_xticks(p_vals)

# 3b: Validite de log^p(n)
ax3b = axes[1]
n_vals = [2, 5, 10, 20, 50, 100]
validity = {
    0: [True]*6,
    1: [True]*6,
    2: [False, True, True, True, True, True],
    3: [False, False, False, True, True, True],
    4: [False]*6
}

for i, p in enumerate([0, 1, 2, 3, 4]):
    y_pos = 4 - i
    for j, n in enumerate(n_vals):
        color = '#2ca02c' if validity[p][j] else '#d62728'
        marker = 'o' if validity[p][j] else 'x'
        ax3b.scatter(j, y_pos, c=color, marker=marker, s=200)

ax3b.set_xticks(range(len(n_vals)))
ax3b.set_xticklabels(n_vals)
ax3b.set_yticks(range(5))
ax3b.set_yticklabels(['p=4', 'p=3', 'p=2', 'p=1', 'p=0'])
ax3b.set_xlabel('n', fontsize=12)
ax3b.set_title('Validite de $\log^{\\circ p}(n) > 0$\n(vert=valide, rouge=invalide)', fontsize=12)

plt.tight_layout()
fig3.savefig(OUTPUT_DIR / "03_p4_artifact.png", dpi=150)
print("[3/6] p4_artifact.png")
plt.close()

# =============================================================================
# Figure 4: Poids log^p(n)
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

n_range = np.arange(2, 101)

for p in [0, 1, 2]:
    weights = []
    for n in n_range:
        val = n
        valid = True
        for _ in range(p):
            if val <= 0:
                valid = False
                break
            val = np.log(val)
        if valid and val > 0:
            weights.append(val)
        else:
            weights.append(np.nan)
    ax4.plot(n_range, weights, label=f'p={p}', linewidth=2)

ax4.set_xlabel('n', fontsize=14)
ax4.set_ylabel('$\log^{\\circ p}(n)$', fontsize=14)
ax4.set_title('Evolution des poids $\log^{\\circ p}(n)$ selon le niveau\n(Les poids s\'aplatissent quand p augmente)', fontsize=14)
ax4.legend(fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(2, 100)

plt.tight_layout()
fig4.savefig(OUTPUT_DIR / "04_weights.png", dpi=150)
print("[4/6] weights.png")
plt.close()

# =============================================================================
# Figure 5: Distance a la ligne critique
# =============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 6))

levels = [0, 1, 2, 3]
distances = []
for p in levels:
    re_vals = [z[0] for z in zeros_by_level[p]]
    dist = np.mean([abs(r - 0.5) for r in re_vals])
    distances.append(dist)

colors_dist = ['#ff7f0e' if d > 0.2 else '#2ca02c' if d < 0.1 else '#1f77b4' for d in distances]
bars = ax5.bar(levels, distances, color=colors_dist, edgecolor='black', linewidth=1.5)

# Annotations
for i, (p, d) in enumerate(zip(levels, distances)):
    ax5.annotate(f'{d:.3f}', (p, d + 0.02), ha='center', fontsize=12, fontweight='bold')

ax5.axhline(y=0, color='green', linestyle='-', linewidth=2)
ax5.set_xlabel('Niveau p', fontsize=14)
ax5.set_ylabel('Distance moyenne a Re(s) = 0.5', fontsize=14)
ax5.set_title('Distance des zeros a la ligne critique\n(p=2 est le plus proche!)', fontsize=14)
ax5.set_xticks(levels)
ax5.grid(True, alpha=0.3, axis='y')

# Highlight p=2
bars[2].set_edgecolor('green')
bars[2].set_linewidth(3)

plt.tight_layout()
fig5.savefig(OUTPUT_DIR / "05_distance_critical.png", dpi=150)
print("[5/6] distance_critical.png")
plt.close()

# =============================================================================
# Figure 6: Resume synthetique
# =============================================================================
fig6, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a: Zeros p=1,2,3 superposes
ax6a = axes[0, 0]
for p in [1, 2, 3]:
    re_vals = [z[0] for z in zeros_by_level[p]]
    im_vals = [z[1] for z in zeros_by_level[p]]
    ax6a.scatter(re_vals, im_vals, c=colors[p], marker=markers[p], s=100, label=f'p={p}', alpha=0.8)
ax6a.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
ax6a.set_xlabel('Re(s)')
ax6a.set_ylabel('Im(s)')
ax6a.set_title('Zeros pour p=1,2,3')
ax6a.legend()
ax6a.grid(True, alpha=0.3)

# 6b: Tendance Re(s) vs p
ax6b = axes[0, 1]
ax6b.plot(levels, mean_re, 'bo-', linewidth=2, markersize=10)
ax6b.fill_between(levels, [m-s for m,s in zip(mean_re, std_re)],
                  [m+s for m,s in zip(mean_re, std_re)], alpha=0.3)
ax6b.axhline(y=0.5, color='red', linestyle='--', linewidth=2)
ax6b.set_xlabel('Niveau p')
ax6b.set_ylabel('Re(s) moyen')
ax6b.set_title('Tendance: Re(s) traverse 0.5 a p~2')
ax6b.set_xticks(levels)
ax6b.grid(True, alpha=0.3)

# 6c: Histogramme de tous les Re(s)
ax6c = axes[1, 0]
all_re = []
for p in [1, 2, 3]:
    all_re.extend([z[0] for z in zeros_by_level[p]])
ax6c.hist(all_re, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax6c.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Re(s)=0.5')
ax6c.axvline(x=np.mean(all_re), color='orange', linestyle='-', linewidth=2, label=f'Moyenne={np.mean(all_re):.2f}')
ax6c.set_xlabel('Re(s)')
ax6c.set_ylabel('Frequence')
ax6c.set_title('Distribution de Re(s) (p=1,2,3)')
ax6c.legend()

# 6d: Tableau de synthese
ax6d = axes[1, 1]
ax6d.axis('off')

summary = """
DECOUVERTES PRINCIPALES

1. FORMULE ZETA INTRINSEQUE:
   zeta_p(s) = exp^{op}(sum (log^{op}(n))^{-(s+1)})

2. CAS p=4 = ARTEFACT:
   - Serie interne VIDE (log^4(n) <= 0)
   - Newton reste bloque sur init (0.5)
   - PAS une confirmation de l'invariance

3. POSITION DES ZEROS (p=1,2,3):
   p=1: Re(s) ~ 0.84  (AU-DESSUS de 0.5)
   p=2: Re(s) ~ 0.43  (PROCHE de 0.5)  <-- SPECIAL!
   p=3: Re(s) ~ 0.30  (EN-DESSOUS de 0.5)

4. CONCLUSIONS:
   - Invariance Re(s)=0.5 NON confirmee
   - Position depend des poids log^p(n)
   - p=2 est un point d'equilibre
   - Hypothese de Riemann peut-etre vraie
     UNIQUEMENT pour p=2
"""

ax6d.text(0.05, 0.95, summary, transform=ax6d.transAxes, fontsize=11,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Synthese des Decouvertes', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig6.savefig(OUTPUT_DIR / "06_summary.png", dpi=150)
print("[6/6] summary.png")
plt.close()

# =============================================================================
# Fin
# =============================================================================
print("\n" + "=" * 70)
print("VISUALISATIONS TERMINEES")
print("=" * 70)
print(f"\nFigures sauvegardees dans: {OUTPUT_DIR.absolute()}")
print("\nFichiers:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {f.name}")

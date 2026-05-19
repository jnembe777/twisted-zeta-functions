"""
Visualisation de la hierarchie des niveaux exponentiels
et comportement des zeros par rapport a RH
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Load data
with open('results/full_experiment/results_latest.json', 'r') as f:
    results = json.load(f)

# Extract hierarchy data
hierarchy = [r for r in results if r['category'] == 'Hierarchy']

# Organize by base and level
data = {}
for base in ['basee', 'base2', 'base10']:
    data[base] = {}
    for r in hierarchy:
        if base in r['name']:
            for part in r['name'].split('_'):
                if 'level' in part:
                    level = int(part.replace('level', ''))
                    data[base][level] = r

# Create figure
fig = plt.figure(figsize=(16, 14))

# ============================================================================
# PLOT 1: Hierarchy diagram with zeros
# ============================================================================
ax1 = fig.add_subplot(2, 2, 1)

levels = [-2, -1, 0, 1, 2]
level_names = {
    -2: 'log(log(...))',
    -1: 'log / racine',
    0: 'Addition (+)',
    1: 'Multiplication (x)',
    2: 'Exponentiation'
}

# Y positions for each base
y_positions = {'basee': 0.7, 'base2': 0.5, 'base10': 0.3}
base_labels = {'basee': 'Base e', 'base2': 'Base 2', 'base10': 'Base 10'}
base_colors = {'basee': '#e74c3c', 'base2': '#3498db', 'base10': '#2ecc71'}

# Draw level boxes and connections
for i, level in enumerate(levels):
    x = i / (len(levels) - 1)

    # Draw box for level
    box_color = '#f0f0f0' if level != 0 else '#ffffcc'
    rect = FancyBboxPatch((x - 0.08, 0.85), 0.16, 0.12,
                          boxstyle="round,pad=0.02",
                          facecolor=box_color, edgecolor='black', linewidth=2)
    ax1.add_patch(rect)

    # Level label
    ax1.text(x, 0.91, f'n = {level}', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax1.text(x, 0.88, level_names[level], ha='center', va='center',
            fontsize=8, style='italic')

    # Draw connections between consecutive levels
    if i < len(levels) - 1:
        ax1.annotate('', xy=((i+1)/(len(levels)-1) - 0.08, 0.91),
                    xytext=(x + 0.08, 0.91),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        # Label "defaut=0"
        mid_x = (x + (i+1)/(len(levels)-1)) / 2
        ax1.text(mid_x, 0.95, 'defaut=0', ha='center', va='bottom',
                fontsize=7, color='green', fontweight='bold')

# Draw zeros for each base
for base, y_base in y_positions.items():
    # Base label
    ax1.text(-0.05, y_base, base_labels[base], ha='right', va='center',
            fontsize=10, fontweight='bold', color=base_colors[base])

    for i, level in enumerate(levels):
        x = i / (len(levels) - 1)

        if level in data[base]:
            r = data[base][level]
            if r['n_zeros'] > 0:
                re_s = r['zeros'][0]['real']
                # Circle size proportional to distance from 0.5
                dist = abs(re_s - 0.5)
                size = 100 + dist * 500

                # Color based on distance
                if dist < 0.05:
                    color = '#27ae60'  # Green - close
                elif dist < 0.15:
                    color = '#f39c12'  # Orange - moderate
                else:
                    color = '#e74c3c'  # Red - far

                ax1.scatter([x], [y_base], s=size, c=color, alpha=0.7,
                           edgecolors='black', linewidth=1.5, zorder=10)
                ax1.text(x, y_base - 0.06, f'{re_s:.3f}', ha='center',
                        fontsize=8, color=base_colors[base])
            else:
                ax1.scatter([x], [y_base], s=50, c='white', alpha=0.5,
                           edgecolors='gray', linewidth=1, marker='x', zorder=10)
                ax1.text(x, y_base - 0.06, 'N/A', ha='center',
                        fontsize=8, color='gray')

# Critical line reference
ax1.axhline(y=0.15, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(1.02, 0.15, 'Re=0.5', ha='left', va='center', fontsize=8, color='red')

ax1.set_xlim(-0.15, 1.1)
ax1.set_ylim(0.1, 1.0)
ax1.axis('off')
ax1.set_title('Hierarchie des Niveaux Exponentiels\n(Taille = distance a Re=0.5, Vert=proche, Rouge=loin)',
             fontsize=12, fontweight='bold')

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
           markersize=10, label='|Re-0.5| < 0.05'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
           markersize=10, label='0.05 < |Re-0.5| < 0.15'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=10, label='|Re-0.5| > 0.15'),
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

# ============================================================================
# PLOT 2: Re(s) vs Level for each base
# ============================================================================
ax2 = fig.add_subplot(2, 2, 2)

for base in ['basee', 'base2', 'base10']:
    levels_plot = []
    re_values = []

    for level in sorted(data[base].keys()):
        r = data[base][level]
        if r['n_zeros'] > 0:
            levels_plot.append(level)
            re_values.append(r['zeros'][0]['real'])

    ax2.plot(levels_plot, re_values, 'o-', color=base_colors[base],
            linewidth=2, markersize=10, label=base_labels[base],
            markeredgecolor='white', markeredgewidth=1.5)

# Critical line
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Ligne critique (RH)')

# Classical arithmetic zone
ax2.axvspan(-0.5, 1.5, alpha=0.1, color='yellow', label='Arithmetique classique')
ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(x=1, color='gray', linestyle=':', alpha=0.5)

ax2.set_xlabel('Niveau n', fontsize=12)
ax2.set_ylabel('Re(s) du zero', fontsize=12)
ax2.set_title('Position des Zeros par Niveau\n(Zone jaune = arithmetique classique)',
             fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xticks(levels)
ax2.set_xticklabels([f'n={l}' for l in levels])
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.6)

# Annotate classical levels
ax2.annotate('Addition', xy=(0, 0.34), xytext=(0, 0.1),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'))
ax2.annotate('Multiplication', xy=(1, 0.3), xytext=(1, 0.05),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

# ============================================================================
# PLOT 3: Distance from critical line vs Level
# ============================================================================
ax3 = fig.add_subplot(2, 2, 3)

for base in ['basee', 'base2', 'base10']:
    levels_plot = []
    distances = []

    for level in sorted(data[base].keys()):
        r = data[base][level]
        if r['n_zeros'] > 0:
            levels_plot.append(level)
            distances.append(abs(r['zeros'][0]['real'] - 0.5))

    ax3.bar([l + {'basee': -0.25, 'base2': 0, 'base10': 0.25}[base] for l in levels_plot],
           distances, width=0.25, color=base_colors[base],
           label=base_labels[base], alpha=0.8, edgecolor='white')

ax3.axhline(y=0, color='green', linestyle='-', linewidth=2)
ax3.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax3.axhline(y=0.15, color='orange', linestyle='--', linewidth=1, alpha=0.5)

ax3.set_xlabel('Niveau n', fontsize=12)
ax3.set_ylabel('Distance a la ligne critique |Re(s) - 0.5|', fontsize=12)
ax3.set_title('Distance des Zeros a Re=0.5 par Niveau',
             fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.set_xticks(levels)
ax3.set_xticklabels([f'n={l}' for l in levels])
ax3.grid(True, alpha=0.3, axis='y')

# Highlight that n=-1 is closest
ax3.annotate('Plus proche\nde RH!', xy=(-1, 0.011), xytext=(-1.5, 0.15),
            fontsize=10, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

# ============================================================================
# PLOT 4: Defect magnitude vs Level (log scale)
# ============================================================================
ax4 = fig.add_subplot(2, 2, 4)

for base in ['basee', 'base2', 'base10']:
    levels_plot = []
    defects = []

    for level in sorted(data[base].keys()):
        r = data[base][level]
        defect = r['defect']['max']
        if defect == float('inf'):
            defect = 1e40
        elif defect == 0:
            defect = 1e-10  # Small value for log scale

        levels_plot.append(level)
        defects.append(defect)

    ax4.semilogy(levels_plot, defects, 'o-', color=base_colors[base],
                linewidth=2, markersize=10, label=base_labels[base],
                markeredgecolor='white', markeredgewidth=1.5)

# Mark zero defect level
ax4.axhline(y=1e-10, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax4.text(2.1, 1e-10, 'Defaut = 0', fontsize=9, color='green', va='center')

ax4.axvline(x=0, color='gold', linestyle='-', linewidth=3, alpha=0.5)
ax4.text(0.05, 1e30, 'n=0\n(ref)', fontsize=9, color='goldenrod')

ax4.set_xlabel('Niveau n', fontsize=12)
ax4.set_ylabel('Defaut maximum (echelle log)', fontsize=12)
ax4.set_title('Magnitude du Defaut par Niveau\n(Defaut explose pour n > 0)',
             fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.set_xticks(levels)
ax4.set_xticklabels([f'n={l}' for l in levels])
ax4.grid(True, alpha=0.3)
ax4.set_ylim(1e-15, 1e45)

# Main title
fig.suptitle('Analyse de la Hierarchie Exponentielle\nComportement des Zeros de la Fonction Zeta Twistee',
            fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('results/hierarchy_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('results/hierarchy_analysis.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Saved: results/hierarchy_analysis.png")
print("Saved: results/hierarchy_analysis.pdf")

# ============================================================================
# FIGURE 2: Detailed transition diagram
# ============================================================================
fig2, ax = plt.subplots(figsize=(14, 8))

# Create a visual flow diagram
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(5, 5.7, 'Transitions entre Niveaux Consecutifs',
       ha='center', fontsize=16, fontweight='bold')
ax.text(5, 5.3, '(Defaut = 0 entre chaque paire consecutive)',
       ha='center', fontsize=11, style='italic', color='green')

# Level boxes
level_x = {-2: 1, -1: 3, 0: 5, 1: 7, 2: 9}
level_info = {
    -2: {'name': 'n = -2', 'op': 'log(log(x))', 're': 'N/A', 'color': '#ffcccc'},
    -1: {'name': 'n = -1', 'op': 'log(x)', 're': '0.511', 'color': '#ccffcc'},
    0: {'name': 'n = 0', 'op': 'x + y', 're': '0.341', 'color': '#ffffcc'},
    1: {'name': 'n = 1', 'op': 'x * y', 're': '0.28-0.36', 'color': '#ffcccc'},
    2: {'name': 'n = 2', 'op': 'exp(log x * log y)', 're': '0.21-1.47', 'color': '#ffcccc'},
}

for level, x in level_x.items():
    info = level_info[level]

    # Box
    rect = FancyBboxPatch((x - 0.8, 2.5), 1.6, 2.2,
                          boxstyle="round,pad=0.05",
                          facecolor=info['color'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Text
    ax.text(x, 4.3, info['name'], ha='center', fontsize=12, fontweight='bold')
    ax.text(x, 3.7, info['op'], ha='center', fontsize=10)
    ax.text(x, 3.2, f"Re(s) = {info['re']}", ha='center', fontsize=10,
           color='blue', fontweight='bold')

    # Distance from 0.5
    if info['re'] != 'N/A' and '-' not in info['re']:
        dist = abs(float(info['re']) - 0.5)
        color = 'green' if dist < 0.05 else ('orange' if dist < 0.15 else 'red')
        ax.text(x, 2.8, f"|Re-0.5| = {dist:.3f}", ha='center', fontsize=9, color=color)

# Arrows between levels
for i, level in enumerate([-2, -1, 0, 1]):
    x1 = level_x[level] + 0.8
    x2 = level_x[level + 1] - 0.8

    ax.annotate('', xy=(x2, 3.5), xytext=(x1, 3.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text((x1 + x2) / 2, 3.8, 'defaut=0', ha='center', fontsize=9,
           color='green', fontweight='bold')

# Highlight classical arithmetic
rect_classic = FancyBboxPatch((4, 1.8), 4, 3.2,
                               boxstyle="round,pad=0.1",
                               facecolor='none', edgecolor='gold',
                               linewidth=4, linestyle='--')
ax.add_patch(rect_classic)
ax.text(6, 1.5, 'ARITHMETIQUE CLASSIQUE', ha='center', fontsize=12,
       fontweight='bold', color='goldenrod')

# Key insight box
insight_text = """OBSERVATION CLE:
Le niveau n=-1 (logarithme) donne Re(s)=0.511
C'est le PLUS PROCHE de la ligne critique Re=0.5 !

Paradoxe: Le niveau 0 (defaut=0) donne Re(s)=0.341
qui est PLUS LOIN de 0.5 que le niveau -1."""

rect_insight = FancyBboxPatch((0.5, 0.2), 4.5, 1.4,
                               boxstyle="round,pad=0.05",
                               facecolor='#e8f4f8', edgecolor='#3498db', linewidth=2)
ax.add_patch(rect_insight)
ax.text(2.75, 0.9, insight_text, ha='center', va='center', fontsize=9,
       fontfamily='monospace')

# RH implication box
rh_text = """IMPLICATION POUR RH:
La coherence algebrique (defaut=0)
ne garantit PAS Re(s) = 0.5

La ligne critique semble etre une
propriete SPECIFIQUE de zeta(s)."""

rect_rh = FancyBboxPatch((5.5, 0.2), 4, 1.4,
                          boxstyle="round,pad=0.05",
                          facecolor='#fff3e0', edgecolor='#e67e22', linewidth=2)
ax.add_patch(rect_rh)
ax.text(7.5, 0.9, rh_text, ha='center', va='center', fontsize=9,
       fontfamily='monospace')

plt.savefig('results/hierarchy_transitions.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Saved: results/hierarchy_transitions.png")

# ============================================================================
# FIGURE 3: Complex plane view of hierarchy zeros
# ============================================================================
fig3, ax = plt.subplots(figsize=(12, 8))

# Critical strip
ax.axvspan(0, 1, alpha=0.1, color='blue', label='Bande critique')
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=3, label='Re = 0.5 (RH)')

# Plot zeros by level with different markers
markers = {-2: 's', -1: '^', 0: 'o', 1: 'D', 2: 'p'}
level_colors = {-2: '#9b59b6', -1: '#27ae60', 0: '#f39c12', 1: '#e74c3c', 2: '#3498db'}

for base in ['basee', 'base2', 'base10']:
    for level in sorted(data[base].keys()):
        r = data[base][level]
        if r['n_zeros'] > 0:
            re_s = r['zeros'][0]['real']
            im_s = r['zeros'][0]['imag']

            ax.scatter([re_s], [im_s], marker=markers[level], s=200,
                      c=level_colors[level], edgecolors='black', linewidth=1.5,
                      alpha=0.8, zorder=10)

            # Add label
            offset = 0.02 if level >= 0 else -0.05
            ax.annotate(f'{base_labels[base]}\nn={level}',
                       xy=(re_s, im_s), xytext=(re_s + offset, im_s + 0.5),
                       fontsize=7, ha='center',
                       arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

# Classical zeta zero
ax.scatter([0.5], [14.1347], marker='*', s=400, c='gold', edgecolors='black',
          linewidth=2, zorder=15, label='1er zero classique')

# Legend for levels
legend_elements = [
    Line2D([0], [0], marker=markers[l], color='w', markerfacecolor=level_colors[l],
           markersize=12, label=f'Niveau {l}') for l in [-2, -1, 0, 1, 2]
]
legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                              markersize=15, label='Zero classique'))

ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_xlabel('Re(s)', fontsize=14)
ax.set_ylabel('Im(s)', fontsize=14)
ax.set_title('Zeros dans le Plan Complexe par Niveau Hierarchique',
            fontsize=14, fontweight='bold')
ax.set_xlim(0.1, 1.6)
ax.set_ylim(4, 22)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/hierarchy_complex_plane.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Saved: results/hierarchy_complex_plane.png")

print("\nAll hierarchy visualizations completed!")

"""
Visualization module for twisted zeta function analysis.

Provides functions for:
- Zero distribution plots in the complex plane
- Defect heatmaps
- Zero trajectory plots under parameter variation
- Chebyshev function plots
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

# Use non-interactive backend by default for server environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches


def plot_zero_distribution(
    zeros: List[complex],
    title: str = "Zero Distribution",
    save_path: Optional[str] = None,
    show_critical_line: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    marker_size: int = 50,
    alpha: float = 0.7
) -> plt.Figure:
    """
    Create a scatter plot of zeros in the complex plane.

    Args:
        zeros: List of complex zeros
        title: Plot title
        save_path: Path to save figure (if None, returns figure)
        show_critical_line: Whether to show Re(s) = 0.5 line
        figsize: Figure size
        marker_size: Size of scatter markers
        alpha: Marker transparency

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if zeros:
        re_parts = [z.real for z in zeros]
        im_parts = [z.imag for z in zeros]

        # Color by distance from critical line
        distances = [abs(z.real - 0.5) for z in zeros]
        scatter = ax.scatter(
            re_parts, im_parts,
            c=distances, cmap='coolwarm',
            s=marker_size, alpha=alpha,
            edgecolors='black', linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Distance from Re(s)=0.5')

    # Critical line
    if show_critical_line:
        y_range = ax.get_ylim()
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2,
                   label='Critical line Re(s)=0.5')
        ax.legend()

    ax.set_xlabel('Re(s)', fontsize=12)
    ax.set_ylabel('Im(s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set reasonable axis limits
    if zeros:
        re_min, re_max = min(re_parts) - 0.1, max(re_parts) + 0.1
        im_min, im_max = min(im_parts) - 1, max(im_parts) + 1
        ax.set_xlim(re_min, re_max)
        ax.set_ylim(im_min, im_max)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_defect_heatmap(
    defect_matrix: 'np.ndarray',
    a_values: List[int],
    b_values: List[int],
    title: str = "Defect Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    symmetric_colorbar: bool = True
) -> plt.Figure:
    """
    Create a heatmap visualization of the defect matrix δ(a,b).

    Args:
        defect_matrix: 2D array of defect values
        a_values: Row labels (a values)
        b_values: Column labels (b values)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap name
        symmetric_colorbar: Center colorbar at zero

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Make colorbar symmetric around zero if requested
    if symmetric_colorbar:
        vmax = np.max(np.abs(defect_matrix))
        vmin = -vmax
    else:
        vmin, vmax = np.min(defect_matrix), np.max(defect_matrix)

    im = ax.imshow(
        defect_matrix, cmap=cmap,
        vmin=vmin, vmax=vmax,
        aspect='auto', origin='lower'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('δ(a,b)', fontsize=12)

    # Set tick labels
    ax.set_xticks(range(len(b_values)))
    ax.set_xticklabels(b_values)
    ax.set_yticks(range(len(a_values)))
    ax.set_yticklabels(a_values)

    ax.set_xlabel('b', fontsize=12)
    ax.set_ylabel('a', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add text annotations for small matrices
    if len(a_values) <= 10 and len(b_values) <= 10:
        for i in range(len(a_values)):
            for j in range(len(b_values)):
                val = defect_matrix[i, j]
                color = 'white' if abs(val) > 0.5 * vmax else 'black'
                ax.text(j, i, f'{val:.2e}', ha='center', va='center',
                        color=color, fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_zero_trajectory(
    trajectory: List[Tuple[float, complex]],
    title: str = "Zero Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    parameter_name: str = "α"
) -> plt.Figure:
    """
    Plot the movement of a zero as a parameter varies.

    Args:
        trajectory: List of (parameter, zero_location) pairs
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        parameter_name: Name of varying parameter

    Returns:
        matplotlib Figure object
    """
    if not trajectory:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center')
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    params = [t[0] for t in trajectory]
    re_parts = [t[1].real for t in trajectory]
    im_parts = [t[1].imag for t in trajectory]

    # Plot in complex plane with color indicating parameter value
    colors = np.linspace(0, 1, len(trajectory))
    scatter = ax1.scatter(
        re_parts, im_parts,
        c=colors, cmap='viridis',
        s=100, alpha=0.8,
        edgecolors='black', linewidths=0.5
    )

    # Connect points with lines
    ax1.plot(re_parts, im_parts, 'k-', alpha=0.3, linewidth=1)

    # Mark start and end
    ax1.scatter([re_parts[0]], [im_parts[0]], color='green', s=200,
                marker='o', label=f'Start ({parameter_name}={params[0]:.3f})')
    ax1.scatter([re_parts[-1]], [im_parts[-1]], color='red', s=200,
                marker='s', label=f'End ({parameter_name}={params[-1]:.3f})')

    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5,
                label='Critical line')
    ax1.set_xlabel('Re(s)', fontsize=12)
    ax1.set_ylabel('Im(s)', fontsize=12)
    ax1.set_title('Zero Location in Complex Plane', fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Colorbar
    sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=params[0], vmax=params[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label(parameter_name, fontsize=10)

    # Plot Re(s) and Im(s) vs parameter
    ax2.plot(params, re_parts, 'b-o', label='Re(s)', markersize=4)
    ax2.plot(params, im_parts, 'r-s', label='Im(s)', markersize=4)
    ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5,
                label='Re(s)=0.5')
    ax2.set_xlabel(parameter_name, fontsize=12)
    ax2.set_ylabel('Zero Component', fontsize=12)
    ax2.set_title(f'Zero Components vs {parameter_name}', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_chebyshev_comparison(
    x_values: List[float],
    psi_phi_values: List[float],
    psi_classical_values: Optional[List[float]] = None,
    title: str = "Twisted Chebyshev Function",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    log_scale: bool = True
) -> plt.Figure:
    """
    Plot twisted Chebyshev function and compare with classical.

    Args:
        x_values: x coordinates
        psi_phi_values: ψ_φ(x) values
        psi_classical_values: Classical ψ(x) values (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        log_scale: Use logarithmic scale for x-axis

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Main plot
    ax1.plot(x_values, psi_phi_values, 'b-', linewidth=2, label='ψ_φ(x)')

    if psi_classical_values:
        ax1.plot(x_values, psi_classical_values, 'r--', linewidth=2,
                 label='ψ(x) classical')

    if log_scale:
        ax1.set_xscale('log')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('ψ(x)', fontsize=12)
    ax1.set_title('Chebyshev Functions', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Ratio plot (if classical values provided)
    if psi_classical_values:
        ratios = [p / c if c != 0 else 0
                  for p, c in zip(psi_phi_values, psi_classical_values)]
        ax2.plot(x_values, ratios, 'g-', linewidth=2)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)

        if log_scale:
            ax2.set_xscale('log')

        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('ψ_φ(x) / ψ(x)', fontsize=12)
        ax2.set_title('Ratio: Twisted / Classical', fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        # Plot growth rate instead
        if len(x_values) > 1:
            growth = [psi_phi_values[i] / x_values[i] for i in range(len(x_values))]
            ax2.plot(x_values, growth, 'g-', linewidth=2)

            if log_scale:
                ax2.set_xscale('log')

            ax2.set_xlabel('x', fontsize=12)
            ax2.set_ylabel('ψ_φ(x) / x', fontsize=12)
            ax2.set_title('Normalized Growth', fontsize=12)
            ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_distribution_defect_growth(
    x_values: List[int],
    delta_values: List[float],
    growth_exponent: Optional[float] = None,
    title: str = "Distribution Defect Growth",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the growth of distribution defect Δ_φ(X).

    Args:
        x_values: X values
        delta_values: Δ_φ(X) values
        growth_exponent: Fitted growth exponent (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Linear-linear plot
    ax1.plot(x_values, delta_values, 'bo-', markersize=8, linewidth=2)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Δ_φ(X)', fontsize=12)
    ax1.set_title('Distribution Defect', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Log-log plot
    valid_mask = np.array(delta_values) > 0
    if np.any(valid_mask):
        x_valid = np.array(x_values)[valid_mask]
        d_valid = np.array(delta_values)[valid_mask]

        ax2.loglog(x_valid, d_valid, 'bo-', markersize=8, linewidth=2,
                   label='Data')

        # Plot fitted line if exponent provided
        if growth_exponent is not None:
            # Fit: Δ = c * X^α
            c = d_valid[0] / (x_valid[0] ** growth_exponent)
            x_fit = np.logspace(np.log10(min(x_valid)), np.log10(max(x_valid)), 100)
            d_fit = c * x_fit ** growth_exponent
            ax2.loglog(x_fit, d_fit, 'r--', linewidth=2,
                       label=f'Fit: X^{growth_exponent:.3f}')
            ax2.legend(fontsize=10)

    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Δ_φ(X)', fontsize=12)
    ax2.set_title('Log-Log Plot', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_zeta_magnitude(
    re_grid: np.ndarray,
    im_grid: np.ndarray,
    zeta_values: np.ndarray,
    title: str = "Magnitude of ζ_φ(s)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    show_zeros: bool = True,
    zero_threshold: float = 0.1
) -> plt.Figure:
    """
    Create a contour/heatmap of |ζ_φ(s)| in the complex plane.

    Args:
        re_grid: Grid of Re(s) values
        im_grid: Grid of Im(s) values
        zeta_values: Complex array of ζ_φ(s) values
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        show_zeros: Highlight regions where |ζ| is small
        zero_threshold: Threshold for zero highlighting

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute magnitude
    magnitude = np.abs(zeta_values)

    # Use log scale for better visualization
    log_magnitude = np.log10(magnitude + 1e-30)

    # Create contour plot
    levels = np.linspace(np.nanmin(log_magnitude), np.nanmax(log_magnitude), 50)
    contour = ax.contourf(re_grid, im_grid, log_magnitude, levels=levels, cmap='viridis')

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('log₁₀|ζ_φ(s)|', fontsize=12)

    # Highlight zeros
    if show_zeros:
        zero_mask = magnitude < zero_threshold
        if np.any(zero_mask):
            ax.contour(re_grid, im_grid, magnitude, levels=[zero_threshold],
                       colors='red', linewidths=2, linestyles='--')

    # Critical line
    ax.axvline(x=0.5, color='white', linestyle='--', linewidth=2,
               label='Critical line')

    ax.set_xlabel('Re(s)', fontsize=12)
    ax.set_ylabel('Im(s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_category_comparison(
    category_data: Dict[str, List[complex]],
    title: str = "Zero Distribution by Category",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Compare zero distributions across different categories.

    Args:
        category_data: Dictionary mapping category names to lists of zeros
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    n_categories = len(category_data)
    if n_categories == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    # Determine grid layout
    cols = min(2, n_categories)
    rows = (n_categories + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_categories == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_categories))

    for idx, (category, zeros) in enumerate(category_data.items()):
        ax = axes[idx]

        if zeros:
            re_parts = [z.real for z in zeros]
            im_parts = [z.imag for z in zeros]

            ax.scatter(re_parts, im_parts, c=[colors[idx]], s=30, alpha=0.6,
                       edgecolors='black', linewidths=0.3)

            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

            # Statistics
            mean_re = np.mean(re_parts)
            std_re = np.std(re_parts)
            on_critical = sum(1 for r in re_parts if abs(r - 0.5) < 0.01)

            stats_text = f'n={len(zeros)}\nmean Re={mean_re:.3f}\nstd Re={std_re:.3f}\non critical={on_critical}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Re(s)', fontsize=10)
        ax.set_ylabel('Im(s)', fontsize=10)
        ax.set_title(category, fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(category_data), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_dashboard(
    transfer_name: str,
    zeros: List[complex],
    defect_matrix: np.ndarray,
    a_values: List[int],
    b_values: List[int],
    chebyshev_data: Optional[Tuple[List[float], List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive dashboard for a single arithmetic transfer.

    Args:
        transfer_name: Name of the transfer
        zeros: List of zeros
        defect_matrix: Defect matrix
        a_values, b_values: Matrix labels
        chebyshev_data: Optional (x_values, psi_values) tuple
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Zero distribution (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if zeros:
        re_parts = [z.real for z in zeros]
        im_parts = [z.imag for z in zeros]
        distances = [abs(z.real - 0.5) for z in zeros]
        scatter = ax1.scatter(re_parts, im_parts, c=distances, cmap='coolwarm',
                              s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax1, label='Dist. from Re=0.5')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Re(s)')
    ax1.set_ylabel('Im(s)')
    ax1.set_title('Zero Distribution')
    ax1.grid(True, alpha=0.3)

    # 2. Defect heatmap (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    vmax = np.max(np.abs(defect_matrix))
    im = ax2.imshow(defect_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax2, label='δ(a,b)')
    ax2.set_xticks(range(len(b_values)))
    ax2.set_xticklabels(b_values, fontsize=8)
    ax2.set_yticks(range(len(a_values)))
    ax2.set_yticklabels(a_values, fontsize=8)
    ax2.set_xlabel('b')
    ax2.set_ylabel('a')
    ax2.set_title('Defect Matrix')

    # 3. Zero histogram (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    if zeros:
        ax3.hist([z.real for z in zeros], bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Re(s)')
    ax3.set_ylabel('Count')
    ax3.set_title('Zero Real Part Distribution')

    # 4. Chebyshev function (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    if chebyshev_data:
        x_vals, psi_vals = chebyshev_data
        ax4.plot(x_vals, psi_vals, 'b-', linewidth=2)
        ax4.set_xscale('log')
        ax4.set_xlabel('x')
        ax4.set_ylabel('ψ_φ(x)')
        ax4.set_title('Twisted Chebyshev Function')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Chebyshev data', ha='center', va='center')

    # 5. Statistics summary (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    stats_lines = [
        f"Transfer: {transfer_name}",
        "",
        "Zero Statistics:",
        f"  Total zeros: {len(zeros)}",
    ]

    if zeros:
        on_critical = sum(1 for z in zeros if abs(z.real - 0.5) < 0.01)
        stats_lines.extend([
            f"  On critical line: {on_critical} ({100*on_critical/len(zeros):.1f}%)",
            f"  Mean Re(s): {np.mean([z.real for z in zeros]):.4f}",
            f"  Std Re(s): {np.std([z.real for z in zeros]):.4f}",
        ])

    stats_lines.extend([
        "",
        "Defect Statistics:",
        f"  Max |δ|: {np.max(np.abs(defect_matrix)):.4e}",
        f"  Mean |δ|: {np.mean(np.abs(defect_matrix)):.4e}",
    ])

    ax5.text(0.1, 0.9, '\n'.join(stats_lines), transform=ax5.transAxes,
             verticalalignment='top', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle(f'Analysis Dashboard: {transfer_name}', fontsize=16, y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

"""
Definition of the 60 arithmetic transfers for the experimental study.

Strata A (Hierarchy, n=15): Iterated exponentials with bases {e, 2, 10} and levels {-2, -1, 0, 1, 2}
Strata B (Coherent Twist, n=15): Bijection-based transfers with vanishing defect
Strata C (Exponential, n=15): φ_α(n) = α^n with α ∈ {0.3, 0.5, ..., 5.0, e, π}
Strata D (Arbitrary, n=30): Polynomials, mixed exponentials, rationals, affines
"""

import math
from typing import List, Tuple, Dict, Any
from mpmath import mp, mpf, exp as mp_exp, log as mp_log, sqrt as mp_sqrt, sinh, asinh, tanh, atanh

from arithmetics.core.transfer import (
    ArithmeticTransfer,
    ExponentialTransfer,
    IteratedExponentialTransfer,
    CoherentTwistTransfer,
    AffineTransfer,
    PolynomialTransfer,
    IdentityTransfer,
    MixedExponentialTransfer,
    RationalTransfer,
)


# Mathematical constants
E = math.e
PI = math.pi


def get_strata_a() -> List[Tuple[ArithmeticTransfer, str]]:
    """
    Generate Strata A: Hierarchy (15 cases).

    Iterated exponential arithmetics with:
    - Bases: {e, 2, 10}
    - Levels: {-2, -1, 0, 1, 2}

    These have vanishing defect by construction.
    """
    bases = [E, 2.0, 10.0]
    base_names = ['e', '2', '10']
    levels = [-2, -1, 0, 1, 2]

    transfers = []

    for base, base_name in zip(bases, base_names):
        for level in levels:
            name = f"Hierarchy_base{base_name}_level{level}"
            phi = IteratedExponentialTransfer(base=base, level=level)
            phi.name = name
            transfers.append((phi, "Hierarchy"))

    return transfers


def get_strata_b() -> List[Tuple[ArithmeticTransfer, str]]:
    """
    Generate Strata B: Coherent Twist (15 cases).

    Bijection-based transfers where operations are twisted coherently,
    resulting in vanishing defect.
    """
    transfers = []

    # 1. log
    transfers.append((
        CoherentTwistTransfer.log_transfer(),
        "CoherentTwist"
    ))
    transfers[-1][0].name = "CoherentTwist_log"

    # 2. exp
    transfers.append((
        CoherentTwistTransfer.exp_transfer(),
        "CoherentTwist"
    ))
    transfers[-1][0].name = "CoherentTwist_exp"

    # 3. square (x²)
    transfers.append((
        CoherentTwistTransfer.square_transfer(),
        "CoherentTwist"
    ))
    transfers[-1][0].name = "CoherentTwist_square"

    # 4. sqrt
    transfers.append((
        CoherentTwistTransfer.sqrt_transfer(),
        "CoherentTwist"
    ))
    transfers[-1][0].name = "CoherentTwist_sqrt"

    # 5. cube (x³)
    transfers.append((
        CoherentTwistTransfer.cube_transfer(),
        "CoherentTwist"
    ))
    transfers[-1][0].name = "CoherentTwist_cube"

    # 6. cbrt (x^{1/3})
    transfers.append((
        CoherentTwistTransfer.cbrt_transfer(),
        "CoherentTwist"
    ))
    transfers[-1][0].name = "CoherentTwist_cbrt"

    # 7. sinh
    phi_sinh = CoherentTwistTransfer(
        g=lambda x: sinh(x),
        g_inverse=lambda y: asinh(y),
        name="sinh"
    )
    phi_sinh.name = "CoherentTwist_sinh"
    transfers.append((phi_sinh, "CoherentTwist"))

    # 8. arcsinh
    phi_arcsinh = CoherentTwistTransfer(
        g=lambda x: asinh(x),
        g_inverse=lambda y: sinh(y),
        name="arcsinh"
    )
    phi_arcsinh.name = "CoherentTwist_arcsinh"
    transfers.append((phi_arcsinh, "CoherentTwist"))

    # 9. tanh (restricted domain)
    phi_tanh = CoherentTwistTransfer(
        g=lambda x: tanh(x),
        g_inverse=lambda y: atanh(y) if mpf(-1) < y < mpf(1) else mpf('nan'),
        name="tanh"
    )
    phi_tanh.name = "CoherentTwist_tanh"
    transfers.append((phi_tanh, "CoherentTwist"))

    # 10. double (x -> 2x)
    phi_double = CoherentTwistTransfer(
        g=lambda x: mpf(2) * x,
        g_inverse=lambda y: y / mpf(2),
        name="double"
    )
    phi_double.name = "CoherentTwist_double"
    transfers.append((phi_double, "CoherentTwist"))

    # 11. half (x -> x/2)
    phi_half = CoherentTwistTransfer(
        g=lambda x: x / mpf(2),
        g_inverse=lambda y: mpf(2) * y,
        name="half"
    )
    phi_half.name = "CoherentTwist_half"
    transfers.append((phi_half, "CoherentTwist"))

    # 12. shift+1 (x -> x + 1, defined on positive reals)
    phi_shift1 = CoherentTwistTransfer(
        g=lambda x: x + mpf(1),
        g_inverse=lambda y: y - mpf(1) if y > mpf(1) else mpf('nan'),
        name="shift_1"
    )
    phi_shift1.name = "CoherentTwist_shift1"
    transfers.append((phi_shift1, "CoherentTwist"))

    # 13. power 1.5 (x -> x^1.5)
    phi_pow15 = CoherentTwistTransfer(
        g=lambda x: x ** mpf('1.5') if x >= 0 else mpf('nan'),
        g_inverse=lambda y: y ** mpf('0.666666666666667') if y >= 0 else mpf('nan'),
        name="power_1.5"
    )
    phi_pow15.name = "CoherentTwist_power1.5"
    transfers.append((phi_pow15, "CoherentTwist"))

    # 14. inverse (x -> 1/x)
    phi_inv = CoherentTwistTransfer(
        g=lambda x: mpf(1) / x if x != 0 else mpf('nan'),
        g_inverse=lambda y: mpf(1) / y if y != 0 else mpf('nan'),
        name="inverse"
    )
    phi_inv.name = "CoherentTwist_inverse"
    transfers.append((phi_inv, "CoherentTwist"))

    # 15. triple (x -> 3x)
    phi_triple = CoherentTwistTransfer(
        g=lambda x: mpf(3) * x,
        g_inverse=lambda y: y / mpf(3),
        name="triple"
    )
    phi_triple.name = "CoherentTwist_triple"
    transfers.append((phi_triple, "CoherentTwist"))

    return transfers


def get_strata_c() -> List[Tuple[ArithmeticTransfer, str]]:
    """
    Generate Strata C: Exponential (15 cases).

    φ_α(n) = α^n with α ∈ {0.3, 0.5, 0.7, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, e, π}

    These have non-trivial defect δ_α(a,b) = α^{a+b} - α^{ab}.
    """
    alphas = [
        0.3, 0.5, 0.7,
        1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        E, PI
    ]

    transfers = []

    for alpha in alphas:
        if alpha == E:
            name = "Exponential_alpha_e"
        elif alpha == PI:
            name = "Exponential_alpha_pi"
        else:
            name = f"Exponential_alpha_{alpha}"

        phi = ExponentialTransfer(alpha=alpha, name=name)
        transfers.append((phi, "Exponential"))

    return transfers


def get_strata_d() -> List[Tuple[ArithmeticTransfer, str]]:
    """
    Generate Strata D: Arbitrary (30 cases).

    Mix of polynomials, mixed exponentials, rationals, and affines.
    """
    transfers = []

    # === Polynomials (10 cases) ===

    # 1. x²
    phi = PolynomialTransfer([0, 0, 1])
    phi.name = "Polynomial_x2"
    transfers.append((phi, "Arbitrary"))

    # 2. x³
    phi = PolynomialTransfer([0, 0, 0, 1])
    phi.name = "Polynomial_x3"
    transfers.append((phi, "Arbitrary"))

    # 3. x² + x
    phi = PolynomialTransfer([0, 1, 1])
    phi.name = "Polynomial_x2_plus_x"
    transfers.append((phi, "Arbitrary"))

    # 4. x² + x + 1
    phi = PolynomialTransfer([1, 1, 1])
    phi.name = "Polynomial_x2_plus_x_plus_1"
    transfers.append((phi, "Arbitrary"))

    # 5. x² + 2x
    phi = PolynomialTransfer([0, 2, 1])
    phi.name = "Polynomial_x2_plus_2x"
    transfers.append((phi, "Arbitrary"))

    # 6. x³ + x²
    phi = PolynomialTransfer([0, 0, 1, 1])
    phi.name = "Polynomial_x3_plus_x2"
    transfers.append((phi, "Arbitrary"))

    # 7. x³ + x
    phi = PolynomialTransfer([0, 1, 0, 1])
    phi.name = "Polynomial_x3_plus_x"
    transfers.append((phi, "Arbitrary"))

    # 8. x² + 1
    phi = PolynomialTransfer([1, 0, 1])
    phi.name = "Polynomial_x2_plus_1"
    transfers.append((phi, "Arbitrary"))

    # 9. x⁴
    phi = PolynomialTransfer([0, 0, 0, 0, 1])
    phi.name = "Polynomial_x4"
    transfers.append((phi, "Arbitrary"))

    # 10. x³ + x² + x
    phi = PolynomialTransfer([0, 1, 1, 1])
    phi.name = "Polynomial_x3_plus_x2_plus_x"
    transfers.append((phi, "Arbitrary"))

    # === Mixed Exponentials (10 cases) ===

    # 11. 2^n + n
    phi = MixedExponentialTransfer(alpha=2.0, beta=1.0, form="alpha^n + beta*n")
    phi.name = "MixedExp_2n_plus_n"
    transfers.append((phi, "Arbitrary"))

    # 12. 2^n * 3^n = 6^n
    phi = MixedExponentialTransfer(alpha=2.0, beta=3.0, form="alpha^n * beta^n")
    phi.name = "MixedExp_2n_times_3n"
    transfers.append((phi, "Arbitrary"))

    # 13. 2^n + 0.5^n
    phi = MixedExponentialTransfer(alpha=2.0, beta=0.5, form="alpha^n + beta^n")
    phi.name = "MixedExp_2n_plus_0.5n"
    transfers.append((phi, "Arbitrary"))

    # 14. 1.5^n + 2n
    phi = MixedExponentialTransfer(alpha=1.5, beta=2.0, form="alpha^n + beta*n")
    phi.name = "MixedExp_1.5n_plus_2n"
    transfers.append((phi, "Arbitrary"))

    # 15. 3^n + 2n
    phi = MixedExponentialTransfer(alpha=3.0, beta=2.0, form="alpha^n + beta*n")
    phi.name = "MixedExp_3n_plus_2n"
    transfers.append((phi, "Arbitrary"))

    # 16. 2.5^n * 1.5^n
    phi = MixedExponentialTransfer(alpha=2.5, beta=1.5, form="alpha^n * beta^n")
    phi.name = "MixedExp_2.5n_times_1.5n"
    transfers.append((phi, "Arbitrary"))

    # 17. 1.2^n + 1.3^n
    phi = MixedExponentialTransfer(alpha=1.2, beta=1.3, form="alpha^n + beta^n")
    phi.name = "MixedExp_1.2n_plus_1.3n"
    transfers.append((phi, "Arbitrary"))

    # 18. 4^n + 0.25n
    phi = MixedExponentialTransfer(alpha=4.0, beta=0.25, form="alpha^n + beta*n")
    phi.name = "MixedExp_4n_plus_0.25n"
    transfers.append((phi, "Arbitrary"))

    # 19. 2^n + 2^n = 2*2^n
    phi = MixedExponentialTransfer(alpha=2.0, beta=2.0, form="alpha^n + beta^n")
    phi.name = "MixedExp_2n_plus_2n"
    transfers.append((phi, "Arbitrary"))

    # 20. 1.5^n * 1.5^n = 2.25^n
    phi = MixedExponentialTransfer(alpha=1.5, beta=1.5, form="alpha^n * beta^n")
    phi.name = "MixedExp_1.5n_times_1.5n"
    transfers.append((phi, "Arbitrary"))

    # === Rational (3 cases) ===

    # 21. n / (n + 1)
    phi = RationalTransfer(c=1.0)
    phi.name = "Rational_n_over_n_plus_1"
    transfers.append((phi, "Arbitrary"))

    # 22. n / (n + 2)
    phi = RationalTransfer(c=2.0)
    phi.name = "Rational_n_over_n_plus_2"
    transfers.append((phi, "Arbitrary"))

    # 23. n / (n + 0.5)
    phi = RationalTransfer(c=0.5)
    phi.name = "Rational_n_over_n_plus_0.5"
    transfers.append((phi, "Arbitrary"))

    # === Affine (6 cases) ===

    # 24. 2n (scaling, zero defect)
    phi = AffineTransfer(c=2.0, d=0.0)
    phi.name = "Affine_2n"
    transfers.append((phi, "Arbitrary"))

    # 25. 3n (scaling, zero defect)
    phi = AffineTransfer(c=3.0, d=0.0)
    phi.name = "Affine_3n"
    transfers.append((phi, "Arbitrary"))

    # 26. 0.5n (scaling, zero defect)
    phi = AffineTransfer(c=0.5, d=0.0)
    phi.name = "Affine_0.5n"
    transfers.append((phi, "Arbitrary"))

    # 27. n + 1 (translation, non-zero defect)
    phi = AffineTransfer(c=1.0, d=1.0)
    phi.name = "Affine_n_plus_1"
    transfers.append((phi, "Arbitrary"))

    # 28. 2n + 1 (affine, non-zero defect)
    phi = AffineTransfer(c=2.0, d=1.0)
    phi.name = "Affine_2n_plus_1"
    transfers.append((phi, "Arbitrary"))

    # 29. n - 1 (translation, non-zero defect)
    phi = AffineTransfer(c=1.0, d=-1.0)
    phi.name = "Affine_n_minus_1"
    transfers.append((phi, "Arbitrary"))

    # === Identity (1 case, control) ===

    # 30. Identity (control case)
    phi = IdentityTransfer()
    phi.name = "Identity_control"
    transfers.append((phi, "Arbitrary"))

    return transfers


def get_all_arithmetics() -> List[Tuple[ArithmeticTransfer, str]]:
    """
    Get all 60 arithmetic transfers.

    Returns:
        List of (transfer, category) tuples
    """
    all_transfers = []

    # Strata A: Hierarchy (15)
    all_transfers.extend(get_strata_a())

    # Strata B: Coherent Twist (15)
    all_transfers.extend(get_strata_b())

    # Strata C: Exponential (15)
    all_transfers.extend(get_strata_c())

    # Strata D: Arbitrary (30) - but we have 30 defined
    all_transfers.extend(get_strata_d())

    return all_transfers


def get_strata(strata_name: str) -> List[Tuple[ArithmeticTransfer, str]]:
    """
    Get transfers for a specific stratum.

    Args:
        strata_name: 'A', 'B', 'C', or 'D'

    Returns:
        List of (transfer, category) tuples for that stratum
    """
    strata_map = {
        'A': get_strata_a,
        'B': get_strata_b,
        'C': get_strata_c,
        'D': get_strata_d,
    }

    if strata_name.upper() not in strata_map:
        raise ValueError(f"Unknown stratum: {strata_name}. Use A, B, C, or D.")

    return strata_map[strata_name.upper()]()


def get_by_category(category: str) -> List[Tuple[ArithmeticTransfer, str]]:
    """
    Get transfers by category name.

    Args:
        category: 'Hierarchy', 'CoherentTwist', 'Exponential', or 'Arbitrary'

    Returns:
        List of (transfer, category) tuples
    """
    all_transfers = get_all_arithmetics()
    return [(phi, cat) for phi, cat in all_transfers if cat == category]


def get_exponential_family(
    alpha_min: float = 0.1,
    alpha_max: float = 5.0,
    n_points: int = 50
) -> List[Tuple[ExponentialTransfer, str]]:
    """
    Generate a family of exponential transfers for parameter study.

    Args:
        alpha_min: Minimum α value
        alpha_max: Maximum α value
        n_points: Number of points

    Returns:
        List of (ExponentialTransfer, 'Exponential') tuples
    """
    import numpy as np
    alphas = np.linspace(alpha_min, alpha_max, n_points)

    # Avoid α = 1
    alphas = [a for a in alphas if abs(a - 1.0) > 0.01]

    transfers = []
    for alpha in alphas:
        phi = ExponentialTransfer(alpha=alpha, name=f"Exp_alpha_{alpha:.4f}")
        transfers.append((phi, "Exponential"))

    return transfers


def summary() -> Dict[str, Any]:
    """
    Generate summary of the population.

    Returns:
        Dictionary with population statistics
    """
    all_transfers = get_all_arithmetics()

    by_category = {}
    for phi, cat in all_transfers:
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(phi.name)

    return {
        'total': len(all_transfers),
        'by_category': {cat: len(names) for cat, names in by_category.items()},
        'categories': list(by_category.keys()),
        'names_by_category': by_category,
    }


def print_population():
    """Print a formatted summary of all arithmetics."""
    all_transfers = get_all_arithmetics()

    print("=" * 70)
    print("TWISTED ZETA FUNCTIONS: POPULATION OF 60 ARITHMETICS")
    print("=" * 70)

    current_category = None
    for i, (phi, category) in enumerate(all_transfers, 1):
        if category != current_category:
            current_category = category
            print(f"\n--- {category} ---")
        print(f"  {i:2d}. {phi.name}")

    print("\n" + "=" * 70)
    summ = summary()
    print(f"Total: {summ['total']} arithmetics")
    for cat, count in summ['by_category'].items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    print_population()

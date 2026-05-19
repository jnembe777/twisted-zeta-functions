"""
Euler product representation for twisted zeta functions.

When unique factorization holds in the twisted monoid (Z_{>0}, ⊗_φ),
the zeta function admits an Euler product over φ-twisted primes:

ζ_φ(s) = ∏_{p ∈ P_φ} (1 - φ(p)^{-s}/p^s)^{-1}
"""

from typing import List, Set, Tuple, Optional, Union, Generator
from mpmath import mp, mpf, mpc, log as mp_log, exp as mp_exp, fabs
import numpy as np

from arithmetics.core.transfer import ArithmeticTransfer, ExponentialTransfer


def sieve_primes(limit: int) -> List[int]:
    """
    Generate classical primes up to limit using Sieve of Eratosthenes.

    Args:
        limit: Upper bound for primes

    Returns:
        List of primes ≤ limit
    """
    if limit < 2:
        return []

    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    return [i for i in range(limit + 1) if is_prime[i]]


def is_twisted_prime(
    n: int,
    phi: ArithmeticTransfer,
    checked_primes: Optional[Set[int]] = None,
    precision: int = 30
) -> bool:
    """
    Check if n is a φ-twisted prime.

    A positive integer p > 1 is a φ-twisted prime if it cannot be written
    as a ⊗_φ b for any 1 < a, b < p.

    Recall: a ⊗_φ b = φ^{-1}(φ(a) · φ(b))

    Args:
        n: Integer to check
        phi: The arithmetic transfer
        checked_primes: Set of known twisted primes (for efficiency)
        precision: Decimal places

    Returns:
        True if n is a φ-twisted prime
    """
    if n <= 1:
        return False

    mp.dps = precision
    phi.precision = precision

    phi_n = phi.phi(n)

    # Check all possible factorizations a ⊗_φ b = n
    # We need φ^{-1}(φ(a) · φ(b)) = n, i.e., φ(a) · φ(b) = φ(n)
    for a in range(2, n):
        phi_a = phi.phi(a)
        if phi_a <= 0:
            continue

        # We need φ(b) = φ(n) / φ(a)
        target_phi_b = phi_n / phi_a

        # Find b such that φ(b) = target_phi_b
        b = phi.phi_inverse(target_phi_b)

        if b is None:
            continue

        # Check if b is a positive integer > 1 and < n
        b_float = float(b)
        if b_float > 1 and b_float < n and abs(b_float - round(b_float)) < 1e-10:
            b_int = round(b_float)
            # Verify: φ(a) · φ(b) ≈ φ(n)
            phi_b = phi.phi(b_int)
            product = phi_a * phi_b
            if fabs(product - phi_n) < mpf(10) ** (-(precision - 5)):
                return False  # n = a ⊗_φ b, so not prime

    return True


def twisted_primes(
    phi: ArithmeticTransfer,
    limit: int,
    precision: int = 30
) -> List[int]:
    """
    Enumerate φ-twisted primes up to limit.

    Args:
        phi: The arithmetic transfer
        limit: Upper bound
        precision: Decimal places

    Returns:
        List of twisted primes ≤ limit
    """
    mp.dps = precision
    phi.precision = precision

    primes = []
    checked = set()

    for n in range(2, limit + 1):
        if is_twisted_prime(n, phi, checked, precision):
            primes.append(n)
            checked.add(n)

    return primes


def twisted_primes_generator(
    phi: ArithmeticTransfer,
    precision: int = 30
) -> Generator[int, None, None]:
    """
    Generator that yields φ-twisted primes indefinitely.

    Args:
        phi: The arithmetic transfer
        precision: Decimal places

    Yields:
        φ-twisted primes in order
    """
    mp.dps = precision
    phi.precision = precision

    checked = set()
    n = 2

    while True:
        if is_twisted_prime(n, phi, checked, precision):
            checked.add(n)
            yield n
        n += 1


def compare_prime_sets(
    phi: ArithmeticTransfer,
    limit: int,
    precision: int = 30
) -> Tuple[List[int], List[int], List[int]]:
    """
    Compare φ-twisted primes with classical primes.

    Args:
        phi: The arithmetic transfer
        limit: Upper bound
        precision: Decimal places

    Returns:
        Tuple (classical_primes, twisted_primes, symmetric_difference)
    """
    classical = set(sieve_primes(limit))
    twisted = set(twisted_primes(phi, limit, precision))

    sym_diff = classical.symmetric_difference(twisted)

    return (
        sorted(classical),
        sorted(twisted),
        sorted(sym_diff)
    )


def euler_product_zeta(
    s: Union[complex, mpc],
    phi: ArithmeticTransfer,
    prime_limit: int = 10000,
    precision: int = 30
) -> mpc:
    """
    Evaluate ζ_φ(s) via Euler product over twisted primes.

    ζ_φ(s) = ∏_{p ∈ P_φ} (1 - φ(p)^{-s}/p^s)^{-1}

    This is only valid when unique factorization holds in (Z_{>0}, ⊗_φ).

    Args:
        s: Complex argument
        phi: The arithmetic transfer
        prime_limit: Upper bound for primes in product
        precision: Decimal places

    Returns:
        ζ_φ(s) computed via Euler product
    """
    mp.dps = precision
    phi.precision = precision

    if isinstance(s, complex):
        s = mpc(s.real, s.imag)

    # Get twisted primes
    primes = twisted_primes(phi, prime_limit, precision)

    # Compute product
    product = mpc(1)

    for p in primes:
        phi_p = phi.phi(p)
        if phi_p <= 0:
            continue

        # Compute (1 - φ(p)^{-s}/p^s)
        log_phi_p = mp_log(phi_p)
        log_p = mp_log(mpf(p))

        # φ(p)^{-s}/p^s = exp(-s(log(φ(p)) + log(p)))
        term_exp = mp_exp(-s * (log_phi_p + log_p))
        factor = 1 - term_exp

        # Invert: (1 - term)^{-1}
        if fabs(factor) > mpf(10) ** (-(precision - 5)):
            product *= 1 / factor
        else:
            # Near singularity (zero of zeta)
            return mpc('inf')

    return product


def euler_product_log_derivative(
    s: Union[complex, mpc],
    phi: ArithmeticTransfer,
    prime_limit: int = 10000,
    precision: int = 30
) -> mpc:
    """
    Compute -ζ'_φ(s)/ζ_φ(s) via Euler product.

    -ζ'/ζ = Σ_{p ∈ P_φ} log(φ(p)·p) · φ(p)^{-s}·p^{-s} / (1 - φ(p)^{-s}·p^{-s})

    Args:
        s: Complex argument
        phi: The arithmetic transfer
        prime_limit: Upper bound for primes
        precision: Decimal places

    Returns:
        -ζ'_φ(s)/ζ_φ(s)
    """
    mp.dps = precision
    phi.precision = precision

    if isinstance(s, complex):
        s = mpc(s.real, s.imag)

    primes = twisted_primes(phi, prime_limit, precision)

    total = mpc(0)

    for p in primes:
        phi_p = phi.phi(p)
        if phi_p <= 0:
            continue

        log_phi_p = mp_log(phi_p)
        log_p = mp_log(mpf(p))

        # x = φ(p)^{-s}·p^{-s}
        x = mp_exp(-s * (log_phi_p + log_p))

        # Term: log(φ(p)·p) · x / (1 - x)
        log_factor = log_phi_p + log_p
        denom = 1 - x

        if fabs(denom) > mpf(10) ** (-(precision - 5)):
            term = log_factor * x / denom
            total += term

    return total


def check_unique_factorization(
    phi: ArithmeticTransfer,
    limit: int = 100,
    precision: int = 30
) -> Tuple[bool, List[Tuple[int, List[Tuple[int, int]]]]]:
    """
    Check if unique factorization holds in (Z_{>0}, ⊗_φ) up to limit.

    Args:
        phi: The arithmetic transfer
        limit: Upper bound for checking
        precision: Decimal places

    Returns:
        Tuple (has_ufd, counterexamples) where counterexamples is
        list of (n, [(a1,b1), (a2,b2), ...]) showing multiple factorizations
    """
    mp.dps = precision
    phi.precision = precision

    primes = twisted_primes(phi, limit, precision)
    prime_set = set(primes)

    counterexamples = []

    for n in range(4, limit + 1):
        factorizations = []

        # Find all factorizations of n as a ⊗_φ b
        phi_n = phi.phi(n)

        for a in range(2, n):
            if a >= n:
                break

            phi_a = phi.phi(a)
            if phi_a <= 0:
                continue

            target_phi_b = phi_n / phi_a
            b = phi.phi_inverse(target_phi_b)

            if b is None:
                continue

            b_float = float(b)
            if 1 < b_float <= n and abs(b_float - round(b_float)) < 1e-10:
                b_int = round(b_float)
                if b_int >= a:  # Avoid duplicates
                    phi_b = phi.phi(b_int)
                    if fabs(phi_a * phi_b - phi_n) < mpf(10) ** (-(precision - 5)):
                        factorizations.append((a, b_int))

        # Check if factorizations are essentially unique
        if len(factorizations) > 1:
            # Filter to keep only factorizations into primes
            prime_factorizations = [
                (a, b) for a, b in factorizations
                if a in prime_set and b in prime_set
            ]
            if len(prime_factorizations) > 1:
                counterexamples.append((n, prime_factorizations))

    has_ufd = len(counterexamples) == 0
    return has_ufd, counterexamples


class TwistedPrimeAnalyzer:
    """
    Comprehensive analyzer for φ-twisted primes and Euler products.
    """

    def __init__(
        self,
        phi: ArithmeticTransfer,
        precision: int = 30
    ):
        """
        Initialize the analyzer.

        Args:
            phi: The arithmetic transfer
            precision: Decimal places
        """
        self.phi = phi
        self.precision = precision
        self._classical_primes = None
        self._twisted_primes = None
        self._ufd_status = None

        mp.dps = precision
        phi.precision = precision

    def get_twisted_primes(self, limit: int) -> List[int]:
        """Get twisted primes up to limit, caching results."""
        if self._twisted_primes is None or len(self._twisted_primes) == 0:
            self._twisted_primes = twisted_primes(self.phi, limit, self.precision)
        elif self._twisted_primes[-1] < limit:
            # Extend
            self._twisted_primes = twisted_primes(self.phi, limit, self.precision)
        return [p for p in self._twisted_primes if p <= limit]

    def get_classical_primes(self, limit: int) -> List[int]:
        """Get classical primes up to limit, caching results."""
        if self._classical_primes is None:
            self._classical_primes = sieve_primes(limit)
        elif len(self._classical_primes) == 0 or self._classical_primes[-1] < limit:
            self._classical_primes = sieve_primes(limit)
        return [p for p in self._classical_primes if p <= limit]

    def prime_comparison(self, limit: int) -> dict:
        """
        Compare classical and twisted primes.

        Args:
            limit: Upper bound

        Returns:
            Dictionary with comparison statistics
        """
        classical = set(self.get_classical_primes(limit))
        twisted = set(self.get_twisted_primes(limit))

        only_classical = classical - twisted
        only_twisted = twisted - classical
        both = classical & twisted

        return {
            'limit': limit,
            'n_classical': len(classical),
            'n_twisted': len(twisted),
            'n_both': len(both),
            'n_only_classical': len(only_classical),
            'n_only_twisted': len(only_twisted),
            'only_classical': sorted(only_classical)[:20],  # First 20
            'only_twisted': sorted(only_twisted)[:20],
            'jaccard_similarity': len(both) / len(classical | twisted) if classical | twisted else 1.0,
        }

    def check_ufd(self, limit: int = 100) -> dict:
        """Check unique factorization property."""
        has_ufd, counterexamples = check_unique_factorization(
            self.phi, limit, self.precision
        )

        self._ufd_status = has_ufd

        return {
            'has_unique_factorization': has_ufd,
            'limit_checked': limit,
            'n_counterexamples': len(counterexamples),
            'counterexamples': counterexamples[:10],  # First 10
        }

    def euler_vs_dirichlet(
        self,
        s: complex,
        prime_limit: int = 1000,
        series_limit: int = 100000
    ) -> dict:
        """
        Compare Euler product and Dirichlet series evaluations.

        Args:
            s: Complex argument
            prime_limit: Limit for Euler product
            series_limit: Terms for Dirichlet series

        Returns:
            Dictionary with comparison results
        """
        from arithmetics.zeta.dirichlet import zeta_phi

        euler_val = euler_product_zeta(s, self.phi, prime_limit, self.precision)
        dirichlet_val = zeta_phi(s, self.phi, series_limit, self.precision)

        diff = fabs(euler_val - dirichlet_val)

        return {
            's': complex(s),
            'euler_product': complex(float(euler_val.real), float(euler_val.imag)),
            'dirichlet_series': complex(float(dirichlet_val.real), float(dirichlet_val.imag)),
            'absolute_difference': float(diff),
            'relative_difference': float(diff / fabs(dirichlet_val)) if fabs(dirichlet_val) > 0 else float(diff),
            'prime_limit': prime_limit,
            'series_limit': series_limit,
        }

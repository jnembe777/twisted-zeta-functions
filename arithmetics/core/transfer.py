"""
ArithmeticTransfer base class and concrete implementations.

An arithmetic transfer is a map φ: (Z,+) → (R_{>0}, ·) that preserves addition
but generally fails to preserve multiplication. The failure is quantified by
the defect δ_φ(a,b) = φ(a)φ(b) - φ(ab).
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple, Union
import math
from mpmath import mp, mpf, exp as mp_exp, log as mp_log, power as mp_power, sqrt as mp_sqrt


class ArithmeticTransfer(ABC):
    """
    Abstract base class for arithmetic transfers.

    An arithmetic transfer φ maps integers to positive reals such that
    φ(a + b) = φ(a) · φ(b) (preserves addition as multiplication).
    """

    def __init__(self, name: str, domain: str = "Z*"):
        """
        Initialize an arithmetic transfer.

        Args:
            name: Human-readable name for this transfer
            domain: Domain specification ("Z*", "Z+", "R+")
        """
        self.name = name
        self.domain = domain
        self._precision = 50  # decimal places for mpmath

    @property
    def precision(self) -> int:
        """Get the current precision in decimal places."""
        return self._precision

    @precision.setter
    def precision(self, value: int):
        """Set the precision for mpmath computations."""
        self._precision = value
        mp.dps = value

    @abstractmethod
    def phi(self, n: Union[int, mpf]) -> mpf:
        """
        Evaluate the transfer function at n.

        Args:
            n: Input value (integer or mpf for high precision)

        Returns:
            φ(n) as an mpf (arbitrary precision float)
        """
        pass

    @abstractmethod
    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """
        Compute the inverse of the transfer function.

        Args:
            y: Value in the range of φ

        Returns:
            n such that φ(n) = y, or None if not invertible
        """
        pass

    def validate_domain(self, n: Union[int, float]) -> bool:
        """
        Check if n is in the domain of this transfer.

        Args:
            n: Value to check

        Returns:
            True if n is in the domain
        """
        if self.domain == "Z*":
            return isinstance(n, (int, mpf)) and n != 0
        elif self.domain == "Z+":
            return isinstance(n, (int, mpf)) and n > 0
        elif self.domain == "R+":
            return n > 0
        return True

    def twisted_multiply(self, a: int, b: int) -> mpf:
        """
        Compute the twisted multiplication a ⊗_φ b.

        Defined as φ^{-1}(φ(a) · φ(b)).

        Args:
            a, b: Integers to multiply

        Returns:
            a ⊗_φ b
        """
        product = self.phi(a) * self.phi(b)
        return self.phi_inverse(product)

    def generator_expr(self) -> str:
        """Return a symbolic expression for this transfer."""
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ExponentialTransfer(ArithmeticTransfer):
    """
    Exponential transfer: φ_α(n) = α^n.

    This is the primary transfer type with non-trivial defect.
    The defect is δ_α(a,b) = α^{a+b} - α^{ab}.
    """

    def __init__(self, alpha: Union[float, mpf], name: Optional[str] = None):
        """
        Initialize an exponential transfer.

        Args:
            alpha: The base of the exponential (α > 0, α ≠ 1)
        """
        self.alpha = mpf(alpha)
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.alpha == 1:
            raise ValueError("alpha = 1 gives trivial transfer")

        name = name or f"Exponential(α={float(alpha):.4g})"
        super().__init__(name, domain="Z*")

    def phi(self, n: Union[int, mpf]) -> mpf:
        """Compute α^n with arbitrary precision."""
        mp.dps = self._precision
        return mp_power(self.alpha, mpf(n))

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """Compute log_α(y) = ln(y) / ln(α)."""
        if y <= 0:
            return None
        mp.dps = self._precision
        return mp_log(mpf(y)) / mp_log(self.alpha)

    def generator_expr(self) -> str:
        return f"α^n where α = {float(self.alpha)}"


class IteratedExponentialTransfer(ArithmeticTransfer):
    """
    Iterated exponential transfer for the hierarchy {⊕_n}.

    For level n:
    - x ⊕_0 y = x + y (addition)
    - x ⊕_1 y = x · y (multiplication)
    - x ⊕_2 y = exp(log(x) · log(y))
    - etc.

    These have vanishing defect by construction.
    """

    def __init__(self, base: Union[float, mpf] = math.e, level: int = 1):
        """
        Initialize an iterated exponential transfer.

        Args:
            base: Base for the exponential (default: e)
            level: Iteration level n ∈ {..., -2, -1, 0, 1, 2, ...}
        """
        self.base = mpf(base)
        self.level = level
        name = f"Iterated(base={float(base):.4g}, level={level})"
        super().__init__(name, domain="R+")

    def _iterated_exp(self, x: mpf, n: int) -> mpf:
        """Compute the n-fold iterated exponential."""
        mp.dps = self._precision
        if n == 0:
            return x
        elif n > 0:
            result = x
            for _ in range(n):
                result = mp_power(self.base, result)
            return result
        else:  # n < 0 (iterated log)
            result = x
            for _ in range(-n):
                if result <= 0:
                    return mpf('nan')
                result = mp_log(result) / mp_log(self.base)
            return result

    def _iterated_log(self, x: mpf, n: int) -> mpf:
        """Compute the n-fold iterated logarithm."""
        return self._iterated_exp(x, -n)

    def phi(self, n: Union[int, mpf]) -> mpf:
        """
        The transfer function for iterated exponential.

        For level k, φ(n) = exp^{(k)}(n).
        """
        mp.dps = self._precision
        return self._iterated_exp(mpf(n), self.level)

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """Inverse is the iterated logarithm."""
        mp.dps = self._precision
        return self._iterated_log(mpf(y), self.level)

    def oplus(self, x: Union[int, float, mpf], y: Union[int, float, mpf]) -> mpf:
        """
        Compute x ⊕_n y = exp^{(n)}(log^{(n)}(x) + log^{(n)}(y)).
        """
        mp.dps = self._precision
        x, y = mpf(x), mpf(y)
        log_x = self._iterated_log(x, self.level)
        log_y = self._iterated_log(y, self.level)
        return self._iterated_exp(log_x + log_y, self.level)

    def generator_expr(self) -> str:
        return f"exp^({self.level})(n) with base {float(self.base)}"


class CoherentTwistTransfer(ArithmeticTransfer):
    """
    Coherent twist transfer: φ_g(x) = g(x) for a bijection g.

    Operations are defined as:
    - x ⊕_g y = g^{-1}(g(x) + g(y))
    - x ⊗_g y = g^{-1}(g(x) · g(y))

    These have vanishing defect by construction when both operations
    are twisted coherently.
    """

    def __init__(
        self,
        g: Callable[[mpf], mpf],
        g_inverse: Callable[[mpf], mpf],
        name: str
    ):
        """
        Initialize a coherent twist transfer.

        Args:
            g: The bijection function
            g_inverse: Inverse of g
            name: Name describing the bijection
        """
        self._g = g
        self._g_inverse = g_inverse
        super().__init__(f"CoherentTwist({name})", domain="R+")

    def phi(self, n: Union[int, mpf]) -> mpf:
        """Apply the bijection g."""
        mp.dps = self._precision
        return self._g(mpf(n))

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """Apply the inverse bijection g^{-1}."""
        mp.dps = self._precision
        try:
            return self._g_inverse(mpf(y))
        except (ValueError, ZeroDivisionError):
            return None

    @classmethod
    def log_transfer(cls) -> 'CoherentTwistTransfer':
        """Create transfer with g = log."""
        return cls(
            g=lambda x: mp_log(x) if x > 0 else mpf('nan'),
            g_inverse=lambda y: mp_exp(y),
            name="log"
        )

    @classmethod
    def exp_transfer(cls) -> 'CoherentTwistTransfer':
        """Create transfer with g = exp."""
        return cls(
            g=lambda x: mp_exp(x),
            g_inverse=lambda y: mp_log(y) if y > 0 else mpf('nan'),
            name="exp"
        )

    @classmethod
    def square_transfer(cls) -> 'CoherentTwistTransfer':
        """Create transfer with g = x^2."""
        return cls(
            g=lambda x: x * x,
            g_inverse=lambda y: mp_sqrt(y) if y >= 0 else mpf('nan'),
            name="x^2"
        )

    @classmethod
    def sqrt_transfer(cls) -> 'CoherentTwistTransfer':
        """Create transfer with g = sqrt(x)."""
        return cls(
            g=lambda x: mp_sqrt(x) if x >= 0 else mpf('nan'),
            g_inverse=lambda y: y * y,
            name="sqrt"
        )

    @classmethod
    def cube_transfer(cls) -> 'CoherentTwistTransfer':
        """Create transfer with g = x^3."""
        return cls(
            g=lambda x: x ** 3,
            g_inverse=lambda y: mp_power(y, mpf('1/3')) if y >= 0 else -mp_power(-y, mpf('1/3')),
            name="x^3"
        )

    @classmethod
    def cbrt_transfer(cls) -> 'CoherentTwistTransfer':
        """Create transfer with g = x^(1/3)."""
        return cls(
            g=lambda x: mp_power(x, mpf('1/3')) if x >= 0 else -mp_power(-x, mpf('1/3')),
            g_inverse=lambda y: y ** 3,
            name="cbrt"
        )

    def generator_expr(self) -> str:
        return f"g(n) where g is {self.name.split('(')[1].rstrip(')')}"


class AffineTransfer(ArithmeticTransfer):
    """
    Affine transfer: φ(x) = c·x + d.

    When d = 0, this reduces to scaling which has trivial defect.
    When d ≠ 0, the defect is non-zero.
    """

    def __init__(self, c: Union[float, mpf], d: Union[float, mpf] = 0):
        """
        Initialize an affine transfer.

        Args:
            c: Scaling coefficient (c ≠ 0)
            d: Translation coefficient
        """
        self.c = mpf(c)
        self.d = mpf(d)
        if self.c == 0:
            raise ValueError("c must be non-zero")

        name = f"Affine(c={float(c)}, d={float(d)})"
        super().__init__(name, domain="Z*")

    def phi(self, n: Union[int, mpf]) -> mpf:
        """Compute c·n + d."""
        mp.dps = self._precision
        return self.c * mpf(n) + self.d

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """Compute (y - d) / c."""
        mp.dps = self._precision
        return (mpf(y) - self.d) / self.c

    def generator_expr(self) -> str:
        if self.d == 0:
            return f"{float(self.c)}·n"
        return f"{float(self.c)}·n + {float(self.d)}"


class PolynomialTransfer(ArithmeticTransfer):
    """
    Polynomial transfer: φ(n) = Σ a_k · n^k.

    General polynomial with arbitrary coefficients.
    """

    def __init__(self, coeffs: List[Union[float, mpf]]):
        """
        Initialize a polynomial transfer.

        Args:
            coeffs: Coefficients [a_0, a_1, ..., a_k] for polynomial
                   φ(n) = a_0 + a_1·n + a_2·n² + ...
        """
        self.coeffs = [mpf(c) for c in coeffs]
        self.degree = len(coeffs) - 1

        # Build name
        terms = []
        for i, c in enumerate(coeffs):
            if c != 0:
                if i == 0:
                    terms.append(f"{float(c)}")
                elif i == 1:
                    terms.append(f"{float(c)}n")
                else:
                    terms.append(f"{float(c)}n^{i}")
        name = f"Polynomial({' + '.join(terms) if terms else '0'})"
        super().__init__(name, domain="Z*")

    def phi(self, n: Union[int, mpf]) -> mpf:
        """Evaluate polynomial at n using Horner's method."""
        mp.dps = self._precision
        n = mpf(n)
        result = mpf(0)
        for c in reversed(self.coeffs):
            result = result * n + c
        return result

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """
        Inverse is not generally available for polynomials of degree > 1.
        For degree 1 (affine), compute directly.
        """
        mp.dps = self._precision
        if self.degree == 0:
            return None  # Constant polynomial has no inverse
        elif self.degree == 1:
            # φ(n) = a_0 + a_1·n, so n = (y - a_0) / a_1
            if self.coeffs[1] == 0:
                return None
            return (mpf(y) - self.coeffs[0]) / self.coeffs[1]
        else:
            # Would need numerical root finding
            return None

    def generator_expr(self) -> str:
        terms = []
        for i, c in enumerate(self.coeffs):
            if c != 0:
                if i == 0:
                    terms.append(f"{float(c)}")
                elif i == 1:
                    terms.append(f"{float(c)}·n")
                else:
                    terms.append(f"{float(c)}·n^{i}")
        return " + ".join(terms) if terms else "0"


class IdentityTransfer(ArithmeticTransfer):
    """
    Identity transfer: φ(n) = n.

    This is the classical case with zero defect.
    Used as a control/reference case.
    """

    def __init__(self):
        super().__init__("Identity", domain="Z*")

    def phi(self, n: Union[int, mpf]) -> mpf:
        """Identity: φ(n) = n."""
        mp.dps = self._precision
        return mpf(n)

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """Identity inverse: φ^{-1}(y) = y."""
        mp.dps = self._precision
        return mpf(y)

    def generator_expr(self) -> str:
        return "n"


class MixedExponentialTransfer(ArithmeticTransfer):
    """
    Mixed exponential transfer: φ(n) = α^n + β·n or similar combinations.
    """

    def __init__(
        self,
        alpha: Union[float, mpf],
        beta: Union[float, mpf],
        form: str = "alpha^n + beta*n"
    ):
        """
        Initialize a mixed exponential transfer.

        Args:
            alpha: Base for exponential term
            beta: Coefficient for linear term
            form: Form of the transfer ("alpha^n + beta*n", "alpha^n * beta^n", etc.)
        """
        self.alpha = mpf(alpha)
        self.beta = mpf(beta)
        self.form = form
        name = f"MixedExp(α={float(alpha)}, β={float(beta)}, form={form})"
        super().__init__(name, domain="Z*")

    def phi(self, n: Union[int, mpf]) -> mpf:
        """Evaluate the mixed exponential."""
        mp.dps = self._precision
        n = mpf(n)

        if self.form == "alpha^n + beta*n":
            return mp_power(self.alpha, n) + self.beta * n
        elif self.form == "alpha^n * beta^n":
            return mp_power(self.alpha, n) * mp_power(self.beta, n)
        elif self.form == "alpha^n + beta^n":
            return mp_power(self.alpha, n) + mp_power(self.beta, n)
        else:
            raise ValueError(f"Unknown form: {self.form}")

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """Inverse generally not available in closed form."""
        return None

    def generator_expr(self) -> str:
        return self.form.replace("alpha", f"{float(self.alpha)}").replace("beta", f"{float(self.beta)}")


class RationalTransfer(ArithmeticTransfer):
    """
    Rational transfer: φ(n) = n / (n + c) or similar.
    """

    def __init__(self, c: Union[float, mpf] = 1):
        """
        Initialize a rational transfer φ(n) = n / (n + c).

        Args:
            c: Constant in denominator
        """
        self.c = mpf(c)
        name = f"Rational(c={float(c)})"
        super().__init__(name, domain="Z+")

    def phi(self, n: Union[int, mpf]) -> mpf:
        """Compute n / (n + c)."""
        mp.dps = self._precision
        n = mpf(n)
        return n / (n + self.c)

    def phi_inverse(self, y: Union[float, mpf]) -> Optional[mpf]:
        """Inverse: n = c·y / (1 - y) for y < 1."""
        mp.dps = self._precision
        y = mpf(y)
        if y >= 1:
            return None
        return self.c * y / (1 - y)

    def generator_expr(self) -> str:
        return f"n / (n + {float(self.c)})"

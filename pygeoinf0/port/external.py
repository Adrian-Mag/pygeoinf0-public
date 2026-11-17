"""Maps given by external code through coordinates (plan 019 F2; experiments/EXP-034-external-implementations/DESIGN.md
2.2).

    External(M, D, g, coords=(C_M, C_D), dom=S)    G(m) = P_D g(E_M m): g is the user's callable on numbers (a ray
                                                   tracer, a PDE solver, ...), C_M and C_D are pg.port.Coordinates of M
                                                   and D (identity coordinates by default for Euclidean spaces)

External is a pg.Map: compositions, partial domains, states and cells take it as they take any map. Two things are
recorded rather than assumed:
- the abstract map G : M -> D and its realisation through coordinates are different objects (a space is not its
  truncation): realises="..." states, with your reason, what the realisation is of; without it the claim is unknown;
- dom is declared. Outside dom the map refuses (pg.Refused); a failure of g at a point of the declared dom is a
  pg.ExternalError naming the code, never a NaN or a skipped datum.

Derivatives (plan 019 F3, DESIGN 2.3): G.derivative(theta) is a LinearOperator M -> D with its Hilbert adjoint, from
one of three providers (x = E_M theta, the numbers at theta):

    jvp=(x, v) -> J v, vjp=(x, w) -> J^T w     products in coordinates (JAX, PyTorch, an adjoint-state code):
                                               DG = P_D J E_M, DG* = E_M* J^T P_D*  (vjp gives the adjoint; without it
                                               the adjoint is refused)
    jacobian=x -> J (k x n)                    the same with a matrix
    measures=x -> [mu_1, ..., mu_k]            the derivative of each datum as a measure on M (a ray path with its
                                               slowness density): DG = P_D Functionals(M.riesz(mu)), no coordinates of M
    linearise=x -> (jvp_x, vjp_x)              one call per model, returning v -> J v and w -> J^T w (vjp_x may be
                                               None) that share their state: a PDE code solves or factors once per
                                               model (plan 020 P2; e.g. SimPEG's fields). The map may be evaluated at
                                               other models between two products (a Taylor test, a line search): state
                                               that the external code keeps in mutable objects must be re-established
                                               by the products (EXP-035 NOTES.md: SimPEG's DC adjoint gap 4.9e-4)

derivative_precision="float32" states that the provider's derivative is computed in single precision (SimPEG's
default sensitivities, an AD tool in float32): a claim (key "precision", holds False), which pg.port.taylor_test cites when
the rate fails.

The adjoints of E_M and P_D carry M's and D's inner products; the provider never sees them. Complex data (plan 020
P3a): jvp, vjp and jacobian may be complex; a vjp returns J^H w (the conjugate transpose), and into a real model's
coordinates the port takes its real part, the real adjoint of a real-linear map. differentiable="..."
asserts, with your reason, that G is differentiable on dom (a provider computes something even across a kink).
"""
import numpy as np

from ..core.claims import ASSERTED, DERIVED, THEOREM, UNKNOWN, Claim
from ..core.errors import ExternalError, Refused
from ..core.logic import Truth
from ..spaces.base import CoordinateSpace
from ..operators.functionals import Functionals
from ..operators.linear import LinearOperator, Matrix
from ..relations.maps import Map
from .coordinates import Coordinates


def _adjoint_numbers(r, Rn):
    """A vjp's numbers J^H w in the domain's coordinates. Into real numbers (a real-linear map from a real model to
    complex data) the real adjoint is Re(J^H w): <J v, w>_R = Re(w^H J v) = v . Re(J^H w) for real v."""
    r = np.asarray(r)
    return r.real if np.iscomplexobj(r) and Rn.field != "C" else r


def _complex_jacobian(J, Rn, Rk):
    """A complex Jacobian as a LinearOperator: v -> J v; adjoint w -> J^H w (its real part into real coordinates)."""
    return LinearOperator(Rn, Rk, lambda v: Rk(J @ v.values),
                          adjoint=lambda w: Rn(_adjoint_numbers(J.conj().T @ w.values, Rn)), name="J")


def _coordinates(space, given, side):
    if given is not None:
        if given.space != space:
            raise ValueError(f"the {side} coordinates are of {given.space}, not of {space}")
        return given
    if isinstance(space, CoordinateSpace):                  # R^n or C^n
        return Coordinates.identity(space)
    raise ValueError(f"{space} has no default coordinates: give the {side} coordinates (pg.port.Coordinates)")


class External(Map):
    """m -> P_D g(E_M m): a map given by external code on numbers (module docstring)."""

    def __init__(self, domain, codomain, g, coords=(None, None), dom=None, name="G", code=None, realises=None,
                 measurable=None, continuous=None, jvp=None, vjp=None, jacobian=None, measures=None,
                 linearise=None, differentiable=None, derivative_precision=None):
        self.coords = (_coordinates(domain, coords[0], "domain"), _coordinates(codomain, coords[1], "codomain"))
        self.g, self.code = g, code or getattr(g, "__name__", "external code")
        given = [k for k, v in (("jvp", jvp), ("jacobian", jacobian), ("measures", measures), ("linearise", linearise))
                 if v is not None]
        if len(given) > 1:
            raise ValueError(f"give one derivative provider, not {' and '.join(given)}")
        if vjp is not None and jvp is None:
            raise ValueError("vjp gives the adjoint of a derivative; give jvp too (or a jacobian), so it can be applied")
        self._provider = {"jvp": jvp, "vjp": vjp, "jacobian": jacobian, "measures": measures, "linearise": linearise}
        self._route = given[0] if given else None
        cin, cout = self.coords

        def f(m):
            x = cin.numbers(m)
            try:
                y = g(x)
            except Exception as e:
                where = "its declared domain" if dom is not None else "the whole domain (no dom was declared)"
                raise ExternalError(f"{self.code} failed inside {where} of {name}: {type(e).__name__}: {e}") from e
            y = np.asarray(y)
            if y.shape != (cout.dim,):
                raise ExternalError(f"{self.code} returned an array of shape {y.shape} for {name}; the codomain's "
                                    f"coordinates have {cout.dim} numbers")
            try:
                return cout.element(y)
            except TypeError as e:                         # e.g. complex numbers from code whose codomain is real
                raise ExternalError(f"{self.code} returned values that {name}'s codomain refuses: {e}") from e

        super().__init__(domain, codomain, f, dom=dom, name=name, measurable=measurable, continuous=continuous)
        statement = f"{name} = P_D {self.code} E_M, a map through coordinates"
        self._claims.append(Claim(statement, ASSERTED, realises, key="realisation") if realises else
                            Claim(statement, UNKNOWN, "what it is a realisation of is not stated (realises=)",
                                  key="realisation"))
        if differentiable:
            self._claims.append(Claim("differentiable", ASSERTED, differentiable, key="differentiable"))
        if derivative_precision is not None and np.dtype(derivative_precision) != np.float64:
            self._claims.append(Claim(f"derivative in {np.dtype(derivative_precision).name}", ASSERTED,
                                      f"stated for {self.code}: relative accuracy about "
                                      f"{np.finfo(np.dtype(derivative_precision)).eps:.0e}", key="precision", holds=False))

    def _call(self, what, fn, *args):
        """A provider's result, with its failures named (pg.ExternalError)."""
        try:
            return fn(*args)
        except Exception as e:
            raise ExternalError(f"{self.code} failed computing the {what} of {self.name}: {type(e).__name__}: {e}") from e

    def derivative(self, theta):
        """DG(theta): a LinearOperator from the domain to the codomain, with its adjoint where the provider gives one
        (module docstring)."""
        if self._route is None:
            raise Refused(f"{self.name} has no derivative provider", needs=("jvp (and vjp), jacobian or measures",))
        if self._dom is not None and self._dom.member(theta) is Truth.FALSE:
            raise Refused(f"{getattr(theta, 'name', 'theta')} is not in dom {self.name}: no derivative there")
        cin, cout = self.coords
        x = cin.numbers(theta)
        Rn, Rk = cin.export.codomain, cout.export.codomain
        p = self._provider
        if self._route == "measures" and Rk.field == "C":
            raise Refused("derivatives given as measures into complex data come with plan 020 P3b")
        if self._route == "measures":
            mus = list(self._call("derivative measures", p["measures"], x))
            if len(mus) != cout.dim:
                raise ExternalError(f"{self.code} gave {len(mus)} derivative measures for {cout.dim} data of {self.name}")
            inner = Functionals(self.domain.riesz(mus), name="Dg")
            how = "the derivative of each datum as a measure on the domain (no coordinates of the domain)"
        else:
            if self._route == "jacobian":
                J = np.asarray(self._call("Jacobian", p["jacobian"], x))
                if J.shape != (cout.dim, cin.dim):
                    raise ExternalError(f"{self.code} gave a Jacobian of shape {J.shape}; {self.name} needs "
                                        f"{(cout.dim, cin.dim)}")
                Jop = Matrix(J, name="J") if not np.iscomplexobj(J) else _complex_jacobian(J, Rn, Rk)
            else:
                if self._route == "linearise":
                    lin = self._call("linearisation", p["linearise"], x)
                    jvp_x, vjp_x = lin
                else:
                    jvp_x = (lambda v: p["jvp"](x, v))
                    vjp_x = None if p["vjp"] is None else (lambda w: p["vjp"](x, w))
                Jop = LinearOperator(Rn, Rk, lambda v: Rk(np.asarray(self._call("JVP", jvp_x, v.values))),
                                     adjoint=None if vjp_x is None else
                                     (lambda w: Rn(_adjoint_numbers(self._call("VJP", vjp_x, w.values), Rn))),
                                     name="J")
            inner = Jop @ cin.export
            how = f"the chain rule P_D J E_M with J from {self._route}"
        D = cout.imp @ inner
        D.name = f"D{self.name}"
        D._claims = [Claim("linear", THEOREM, "a composition of linear maps", key="linear"),
                     Claim(f"the derivative of {self.name} at {getattr(theta, 'name', 'theta')}", ASSERTED,
                           f"supplied by {self.code}: {how}", key="derivative"),
                     Claim("adjoint in the inner products of the domain and the codomain", DERIVED,
                           "the adjoints of the coordinate maps are the Riesz maps of the two spaces", key="adjoint")]
        return D

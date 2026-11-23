"""Complete-spectrum warm starts; no current eigensystem enters update().

The optional fixed incidence/weight interface pays for rank-one edge updates.
All outputs are full n-by-n U and n eigenvalues.  Both candidate methods pay
for actual residual and orthogonality guards, and fall back to full eigh.
Numba compilation is setup, not a steady-state timing observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from scipy.linalg import qr, eigh

TOLERANCE = 2e-6
ORTHOGONALITY_TOLERANCE = 1e-10
METHODS = ("warm_jacobi", "paired_qr")
BASELINES = ("exact_eigh", "exact_evr")


@njit(cache=True)
def _validate_laplacian(lap):
    scale2, asymmetric2 = 0.0, 0.0
    finite = True
    for i in range(len(lap)):
        for j in range(len(lap)):
            value = lap[i, j]
            finite = finite and np.isfinite(value)
            scale2 += value * value
            asymmetric2 += (value - lap[j, i]) ** 2
    return finite, np.sqrt(scale2), np.sqrt(asymmetric2)


@njit(cache=True)
def _validate_edge_model(lap, starts, ends, weights):
    expected = np.zeros_like(lap)
    for edge in range(len(weights)):
        value = weights[edge]
        if not np.isfinite(value) or value < 0.0:
            return False
        i, j = starts[edge], ends[edge]
        expected[i, i] += value; expected[j, j] += value
        expected[i, j] -= value; expected[j, i] -= value
    for i in range(len(lap)):
        for j in range(len(lap)):
            if abs(lap[i, j] - expected[i, j]) > 2e-13 + 2e-13 * abs(expected[i, j]):
                return False
    return True


@njit(cache=True)
def _certify(lap, values, u):
    """Paid full actual residual and orthogonality, with fused small loops."""
    n = len(lap)
    residual2, orth2, scale2 = 0.0, 0.0, 0.0
    for i in range(n):
        for j in range(n):
            scale2 += lap[i, j] * lap[i, j]
            applied, inner = 0.0, 0.0
            for k in range(n):
                applied += lap[i, k] * u[k, j]
                inner += u[k, i] * u[k, j]
            residual2 += (applied - u[i, j] * values[j]) ** 2
            if i == j:
                inner -= 1.0
            orth2 += inner * inner
    return np.sqrt(residual2) / max(np.sqrt(scale2), 1e-30), np.sqrt(orth2)


@njit(cache=True)
def _edge_increment(a, u, starts, ends, delta):
    """A <- A + sum_e delta_e z_e z_e^T, z_e=U[start]-U[end]."""
    n = u.shape[0]
    z = np.empty(n)
    for edge in range(len(delta)):
        change = delta[edge]
        if change == 0.0:
            continue
        for i in range(n):
            z[i] = u[starts[edge], i] - u[ends[edge], i]
        for i in range(n):
            for j in range(i, n):
                a[i, j] += change * z[i] * z[j]
                a[j, i] = a[i, j]
    return a


@njit(cache=True)
def _edge_apply(u, starts, ends, weights):
    result = np.zeros_like(u)
    n = u.shape[0]
    for edge in range(len(weights)):
        start, end = starts[edge], ends[edge]
        for j in range(n):
            value = weights[edge] * (u[start, j] - u[end, j])
            result[start, j] += value
            result[end, j] -= value
    return result


@njit(cache=True)
def _rotate_pair(a, u, p, q):
    """Stable exact Jacobi plane solve; updates A and every row of U."""
    apq = a[p, q]
    if apq == 0.0:
        return 0
    app, aqq = a[p, p], a[q, q]
    tau = (aqq - app) / (2.0 * apq)
    if tau >= 0.0:
        tangent = 1.0 / (tau + np.hypot(1.0, tau))
    else:
        tangent = -1.0 / (-tau + np.hypot(1.0, tau))
    cosine = 1.0 / np.sqrt(1.0 + tangent * tangent)
    sine = tangent * cosine
    n = a.shape[0]
    for k in range(n):
        if k != p and k != q:
            akp, akq = a[k, p], a[k, q]
            a[k, p] = cosine * akp - sine * akq
            a[p, k] = a[k, p]
            a[k, q] = sine * akp + cosine * akq
            a[q, k] = a[k, q]
        ukp, ukq = u[k, p], u[k, q]
        u[k, p] = cosine * ukp - sine * ukq
        u[k, q] = sine * ukp + cosine * ukq
    a[p, p] = app - tangent * apq
    a[q, q] = aqq + tangent * apq
    a[p, q], a[q, p] = 0.0, 0.0
    return 1


@njit(cache=True)
def _off_norm(a):
    total = 0.0
    for p in range(len(a)):
        for q in range(p + 1, len(a)):
            total += 2.0 * a[p, q] * a[p, q]
    return np.sqrt(total)


@njit(cache=True)
def _cyclic_jacobi(a, u, target, max_sweeps):
    n = len(a)
    rotations = 0
    # This threshold has a deterministic Frobenius residual bound. A sweep
    # can create new couplings; each completed sweep is checked again.
    threshold = target / np.sqrt(max(1, n * (n - 1)))
    for sweep in range(max_sweeps):
        if _off_norm(a) <= target:
            return a, u, sweep, rotations, True
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p, q]) > threshold:
                    rotations += _rotate_pair(a, u, p, q)
    return a, u, max_sweeps, rotations, _off_norm(a) <= target


@njit(cache=True)
def _paired_rotations(a, u, pairs):
    count = 0
    for k in range(len(pairs)):
        count += _rotate_pair(a, u, pairs[k, 0], pairs[k, 1])
    return a, u, count


def partition(n):
    if n < 4 or n % 2:
        raise ValueError("The fixed paired partition requires an even n >= 4")
    return [np.array([0])] + [np.array([j, j + 1]) for j in range(1, n - 1, 2)] + [np.array([n - 1])]


def _incidence_edges(b):
    b = np.asarray(b, dtype=float)
    starts, ends = [], []
    for column in b.T:
        plus = np.flatnonzero(column == 1.0)
        minus = np.flatnonzero(column == -1.0)
        if len(plus) != 1 or len(minus) != 1 or np.count_nonzero(column) != 2:
            raise ValueError("Edge mode requires oriented +/-1 incidence columns")
        starts.append(plus[0]); ends.append(minus[0])
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def diagnostics(lap, values, u):
    """Public, identically priced full-spectrum baseline/candidate guards."""
    residual, orth = _certify(np.asarray(lap, dtype=float),
                              np.asarray(values, dtype=float), np.asarray(u, dtype=float))
    return dict(actual_residual=float(residual), orthogonality_error=float(orth))


def make_contract(b):
    """One-time fixed graph validation; return cached integer edge endpoints."""
    return _incidence_edges(b)


def validate_contract(lap, weights=None, edges=None):
    """Public paid input checks; return the Frobenius norm of actual L."""
    lap = np.asarray(lap, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise ValueError("A finite square Laplacian is required")
    finite, norm, asymmetry = _validate_laplacian(lap)
    if not finite or asymmetry > 1e-12 * max(1.0, norm):
        raise ValueError("A finite symmetric Laplacian is required")
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if edges is None or w.shape != (len(edges[0]),):
            raise ValueError("Matching edge endpoints and weights required")
        if not _validate_edge_model(lap, *edges, w):
            raise ValueError("Nonnegative weights and matching Laplacian required")
    return float(norm)


@dataclass
class SpectrumState:
    values: np.ndarray
    vectors: np.ndarray
    rayleigh: np.ndarray | None
    laplacian: np.ndarray
    weights: np.ndarray | None = None


class SpectrumTracker:
    """initialize/update return ((values, vectors), diagnostics).

    Pass incidence at construction and weights at each call to activate the
    edge cache. 'laplacian' must be exactly the supplied incidence/weights
    model; initialize checks this contract once, update checks shape/finite
    weights and then uses that model for the paid actual residual. The model
    is fixed, not discovered from current exact eigenvectors.
    """
    def __init__(self, mode="warm_jacobi", incidence=None, groups=None,
                 max_sweeps=6, max_newton_steps=3,
                 tolerance=TOLERANCE, orthogonality_tolerance=ORTHOGONALITY_TOLERANCE):
        if mode not in METHODS + BASELINES:
            raise ValueError(mode)
        if tolerance != TOLERANCE or orthogonality_tolerance != ORTHOGONALITY_TOLERANCE:
            raise ValueError("Development tolerances are frozen")
        self.mode = mode
        self.incidence = None if incidence is None else np.asarray(incidence, dtype=float).copy()
        self.edges = None if incidence is None else _incidence_edges(self.incidence)
        self.groups = groups
        self.max_sweeps, self.max_newton_steps = max_sweeps, max_newton_steps
        self.state = None

    def _inputs(self, laplacian, weights):
        lap = np.asarray(laplacian, dtype=float)
        w = None
        if weights is not None:
            if self.edges is None:
                raise ValueError("weights require incidence")
            w = np.asarray(weights, dtype=float)
            if w.shape != (self.incidence.shape[1],):
                raise ValueError("Nonnegative finite edge weights required")
            if lap.shape[0] != self.incidence.shape[0]:
                raise ValueError("Incidence dimension mismatch")
        norm = validate_contract(lap, w, self.edges)
        return lap, w, norm

    def _eigh(self, lap):
        if self.mode == "exact_evr":
            return eigh(lap, driver="evr", check_finite=False)
        return np.linalg.eigh(lap)

    def initialize(self, laplacian, weights=None):
        lap, w, _ = self._inputs(laplacian, weights)
        values, u = self._eigh(lap)
        # Cache the actual Rayleigh matrix, not the diagonal approximation.
        a = None if self.mode in BASELINES else u.T @ lap @ u
        self.state = SpectrumState(values, u, a, lap.copy(), None if w is None else w.copy())
        return (values, u), self._metadata(lap, values, u, "initial", 0, 0, 0, "initial_evd")

    def _metadata(self, lap, values, u, reason, sweeps, rotations, block_solves, cache_mode):
        # The actual supplied Laplacian is used, including edge-mode calls.
        checked = diagnostics(lap, values, u)
        return dict(refresh=int(reason != "none"), refresh_reason=reason,
                    **checked,
                    sweeps=sweeps, rotation_steps=rotations,
                    block_solves=block_solves, cached_residual=0.0,
                    cache_mode=cache_mode)

    def update(self, laplacian, weights=None):
        if self.state is None:
            return self.initialize(laplacian, weights)
        lap, w, norm = self._inputs(laplacian, weights)
        previous = self.state
        if lap.shape != previous.laplacian.shape:
            raise ValueError("The graph dimension must stay fixed")
        if self.mode in BASELINES:
            values, u = self._eigh(lap)
            meta = self._metadata(lap, values, u, "exact_each_step", 0, 0, 0, "none")
            if meta["actual_residual"] > TOLERANCE or meta["orthogonality_error"] > ORTHOGONALITY_TOLERANCE:
                raise ArithmeticError("Exact EVD failed the common numerical guard")
            # A full EVD baseline has no use for a warm-start Rayleigh cache.
            self.state = SpectrumState(values, u, None, lap.copy(), None if w is None else w.copy())
            return (values, u), meta
        u = previous.vectors.copy()
        if w is not None and previous.weights is not None:
            a = _edge_increment(previous.rayleigh.copy(), u, *self.edges, w - previous.weights)
            cache_mode = "edge_rank_one"
        else:
            a = previous.rayleigh + u.T @ (lap - previous.laplacian) @ u
            a = (a + a.T) * .5
            cache_mode = "dense_increment"
        norm = max(float(norm), 1e-30)
        # Aim below the unchanged acceptance threshold; this margin accounts
        # only for roundoff between cached and actual Rayleigh calculations.
        target = .8 * TOLERANCE * norm
        reason = "none"
        sweeps = rotations = block_solves = 0
        if self.mode == "warm_jacobi":
            a, u, sweeps, rotations, converged = _cyclic_jacobi(a, u, target, self.max_sweeps)
            if not converged:
                reason = "jacobi_sweep_limit"
        else:
            groups = self.groups or partition(len(lap))
            labels = np.empty(len(lap), dtype=int)
            pairs = []
            for index, group in enumerate(groups):
                labels[group] = index
                if len(group) == 2:
                    pairs.append(group)
                elif len(group) != 1:
                    raise ValueError("paired_qr requires singleton/pair clusters")
            pairs = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
            cross = labels[:, None] != labels[None, :]
            for iteration in range(self.max_newton_steps + 1):
                a, u, count = _paired_rotations(a, u, pairs)
                block_solves += count
                values = np.diag(a).copy()
                off = a - np.diag(values)
                if float(np.linalg.norm(off, "fro")) <= target:
                    break
                if iteration == self.max_newton_steps:
                    reason = "qr_iteration_limit"; break
                gaps = values[None, :] - values[:, None]
                if np.min(np.abs(gaps[cross])) < 1e-5 * max(1.0, np.max(np.abs(values))):
                    reason = "small_external_gap"; break
                omega = np.zeros_like(a)
                omega[cross] = off[cross] / gaps[cross]
                omega = (omega - omega.T) * .5
                # Frobenius is a conservative bound and needs no extra SVD.
                if np.linalg.norm(omega, "fro") > .35:
                    reason = "large_rotation"; break
                q, r = qr(np.eye(len(lap)) + omega, mode="economic", check_finite=False)
                signs = np.sign(np.diag(r)); signs[signs == 0] = 1.0
                q *= signs
                u = u @ q
                a = q.T @ a @ q
                a = (a + a.T) * .5
                rotations += 1
        values = np.diag(a).copy()
        order = np.argsort(values, kind="stable")
        values, u, a = values[order], u[:, order], a[np.ix_(order, order)]
        meta = self._metadata(lap, values, u, reason, sweeps, rotations, block_solves, cache_mode)
        meta["cached_residual"] = float(_off_norm(a) / norm)
        if reason == "none" and meta["orthogonality_error"] > ORTHOGONALITY_TOLERANCE:
            reason = "orthogonality_guard"
        if reason == "none" and meta["actual_residual"] > TOLERANCE * (1 + 1e-8):
            reason = "actual_residual_guard"
        if reason != "none":
            values, u = np.linalg.eigh(lap)
            a = u.T @ lap @ u
            meta = self._metadata(lap, values, u, reason, sweeps, rotations, block_solves, cache_mode)
            meta["cached_residual"] = float(_off_norm(a) / norm)
        self.state = SpectrumState(values, u, a, lap.copy(), None if w is None else w.copy())
        return (values, u), meta


def compile_kernels():
    """Pay and return one-time JIT wall seconds; never hide this in timing."""
    start = time.perf_counter()
    a = np.diag(np.arange(4, dtype=float)); a[1, 2] = a[2, 1] = .001
    u = np.eye(4); starts = np.array([0, 1], dtype=np.int64); ends = np.array([1, 2], dtype=np.int64)
    _validate_laplacian(a)
    _validate_edge_model(a, starts, ends, np.array([1., 1.]))
    for layout in ("C", "F"):
        frame = np.array(u, order=layout)
        _edge_increment(a.copy(), frame.copy(order=layout), starts, ends, np.array([.001, .002]))
        _edge_apply(frame, starts, ends, np.array([1., 1.]))
        _certify(a, np.diag(a).copy(), frame)
        _cyclic_jacobi(a.copy(), frame.copy(order=layout), 1e-7, 6)
        _paired_rotations(a.copy(), frame.copy(order=layout), np.array([[1, 2]], dtype=np.int64))
    _off_norm(a)
    return time.perf_counter() - start


def step(method, lap, previous, t, groups, *, b=None, weights=None):
    """Runner adapter: return ((values, vectors, tracker), diagnostics).

    The first two state entries match the old baseline. The third is this
    method's own cache and must be returned unchanged as 'previous'.
    initialize is full eigh; later calls only use this cache and current L/w.
    """
    if previous is None:
        tracker = SpectrumTracker(method, incidence=b, groups=groups)
        result, meta = tracker.initialize(lap, weights)
    else:
        if len(previous) != 3 or not isinstance(previous[2], SpectrumTracker):
            raise ValueError("Pass this method's previous three-entry state")
        tracker = previous[2]
        if tracker.mode != method:
            raise ValueError("A spectrum cache cannot be shared across methods")
        result, meta = tracker.update(lap, weights)
    return (result[0], result[1], tracker), meta

"""Complete spectra for a shared, observation-inferred nonnegative graph.

No truth, reference spectrum, or true switch time enters SpectrumPair.  The
reference-only metrics below are deliberately outside the timed step routine.
The old warm-Jacobi implementation is copied byte-for-byte under vendor/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from vendor import spectrum_fast as _old

METHODS = ("exact_eigh", "exact_evr", "warm_jacobi", "periodic2", "periodic5", "residual_refresh")
RESIDUAL_TOL = 2e-6
ORTH_TOL = 1e-10
VENDOR_SHA256 = "b4c253d9c187dc0383f21637601730965902b3fbf29a2395bd928fc60c40ce7e"


def make_laplacian(B, theta):
    """Build L=B diag(theta) B.T; exact zeros represent absent edges."""
    B = np.asarray(B, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if B.ndim != 2 or theta.shape != (B.shape[1],):
        raise ValueError("Incidence and edge-weight dimensions do not match")
    if not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise ValueError("Finite, nonnegative edge weights are required")
    return (B * theta) @ B.T


class SpectrumPair:
    """Six separately cached complete-spectrum methods for one inferred graph.

    Call setup() once, outside a formal steady-state timing interval.  step()
    accepts a common L optionally constructed/timed by the outer runner.
    Output is {method: {values, vectors, meta, elapsed_s}}.  Each method pays
    its input check, core certificate, and an identical final certificate;
    unsuccessful attempts and fallback costs remain inside elapsed_s.
    Periodic methods intentionally retain uncertified approximations between
    scheduled refreshes; their numerical_pass flag must be honored.
    """
    def __init__(self, n, B):
        self.n = int(n)
        self.B = np.array(B, dtype=float, copy=True)
        if self.B.ndim != 2 or self.B.shape[0] != self.n:
            raise ValueError("Incidence row count must equal n")
        self.edges = _old.make_contract(self.B)
        self.trackers = {
            method: _old.SpectrumTracker(mode=method, incidence=self.B)
            for method in ("exact_eigh", "exact_evr", "warm_jacobi")
        }
        self.cache = {}
        self.t = 0
        self.setup_info = None
        self.last_laplacian = None

    def setup(self):
        started = time.perf_counter()
        copied_source = Path(_old.__file__)
        actual_hash = hashlib.sha256(copied_source.read_bytes()).hexdigest()
        if actual_hash != VENDOR_SHA256:
            raise RuntimeError("Vendored source hash differs from sealed implementation")
        jit_seconds = _old.compile_kernels()
        # Warm both dense libraries and common guard/layout signatures.  These
        # small matrices never become trajectory caches or observation inputs.
        L = np.diag([0., 1., 2., 3.])
        for solver in (np.linalg.eigh, lambda x: eigh(x, driver="evr", check_finite=False)):
            values, vectors = solver(L)
            _old.diagnostics(L, values, vectors)
        self.setup_info = {
            "setup_s": time.perf_counter() - started,
            "jit_s": jit_seconds,
            "vendor_sha256": actual_hash,
            "timing_contract": "steady-state excludes setup, common L construction, public evaluation and persistence",
            "common_certificates_per_successful_nontrigger_step": 2,
        }
        return dict(self.setup_info)

    def _baseline(self, method, lap, theta):
        # The same paid input validation used inside the vendored trackers.
        _old.validate_contract(lap, theta, self.edges)
        previous = self.cache.get(method)
        reason = "none"
        attempted_residual = None
        attempted_orth = None
        scheduled = method.startswith("periodic") and self.t % int(method.removeprefix("periodic")) == 0
        if previous is None or scheduled:
            values, vectors = np.linalg.eigh(lap)
            reason = "initial" if previous is None else "scheduled"
            cert = _old.diagnostics(lap, values, vectors)
        else:
            vectors = previous[1]
            # Preserve the full previous basis and recompute the complete
            # Rayleigh diagonal, including a paid matrix application.
            values = np.einsum("ij,ij->j", vectors, lap @ vectors)
            order = np.argsort(values, kind="stable")
            values, vectors = values[order], vectors[:, order]
            cert = _old.diagnostics(lap, values, vectors)
            attempted_residual = cert["actual_residual"]
            attempted_orth = cert["orthogonality_error"]
            if method == "residual_refresh" and attempted_residual > RESIDUAL_TOL:
                values, vectors = np.linalg.eigh(lap)
                reason = "residual_event"
                cert = _old.diagnostics(lap, values, vectors)
            elif attempted_orth > ORTH_TOL:
                values, vectors = np.linalg.eigh(lap)
                reason = "orthogonality_guard"
                cert = _old.diagnostics(lap, values, vectors)
        meta = dict(
            refresh=int(reason != "none"), refresh_reason=reason,
            rotation_steps=0, sweeps=0, block_solves=0,
            cache_mode="full_previous_basis", scheduled_refresh=bool(scheduled),
            attempted_residual=attempted_residual,
            attempted_orthogonality=attempted_orth,
            **cert,
        )
        return values, vectors, meta

    def _strict_final(self, method, lap, theta, values, vectors, meta):
        # One additional common actual-matrix certificate is paid by every
        # method, including exact EVD.  Its stricter <= condition closes the
        # tiny acceptance margin in the unmodified historical warm tracker.
        final = _old.diagnostics(lap, values, vectors)
        passed = final["actual_residual"] <= RESIDUAL_TOL and final["orthogonality_error"] <= ORTH_TOL
        if not passed and not method.startswith("periodic"):
            meta["pre_wrapper_refresh_reason"] = meta["refresh_reason"]
            meta["pre_wrapper_residual"] = final["actual_residual"]
            values, vectors = np.linalg.eigh(lap)
            meta["refresh"] = 1
            meta["refresh_reason"] = "strict_common_guard"
            if method in self.trackers:
                tracker = self.trackers[method]
                rayleigh = vectors.T @ lap @ vectors if method == "warm_jacobi" else None
                tracker.state = _old.SpectrumState(values, vectors, rayleigh, lap.copy(), theta.copy())
            final = _old.diagnostics(lap, values, vectors)
            passed = final["actual_residual"] <= RESIDUAL_TOL and final["orthogonality_error"] <= ORTH_TOL
        meta.update(final)
        meta["numerical_pass"] = bool(passed)
        meta["full_eigenvalues"] = self.n
        meta["full_eigenvector_columns"] = self.n
        meta["time_index"] = self.t
        return values, vectors, meta

    def step(self, theta, laplacian=None):
        theta = np.asarray(theta, dtype=float)
        # Common input graph construction sits outside individual method times
        # and must be charged once by the caller in an end-to-end total.
        lap = make_laplacian(self.B, theta) if laplacian is None else np.asarray(laplacian, dtype=float)
        self.last_laplacian = lap
        result = {}
        for method in METHODS:
            started = time.perf_counter()
            if method in self.trackers:
                (values, vectors), meta = self.trackers[method].update(lap, theta)
            else:
                values, vectors, meta = self._baseline(method, lap, theta)
            values, vectors, meta = self._strict_final(method, lap, theta, values, vectors, meta)
            # Persist only this method's own complete result as its next cache.
            self.cache[method] = (values, vectors)
            result[method] = dict(values=values, vectors=vectors, meta=meta,
                                  elapsed_s=time.perf_counter() - started)
        self.t += 1
        return result


def numerical_metrics(L, values, U, refvalues, refU):
    """Reference-only evaluation; never called inside a candidate update."""
    L = np.asarray(L, dtype=float)
    values = np.asarray(values, dtype=float)
    U = np.asarray(U, dtype=float)
    refvalues = np.asarray(refvalues, dtype=float)
    refU = np.asarray(refU, dtype=float)
    norm = max(float(np.linalg.norm(L, "fro")), 1e-30)
    cert = _old.diagnostics(L, values, U)
    gap = float(refvalues[2] - refvalues[1]) if len(values) > 2 else float("nan")
    isolated = len(values) > 2 and gap > 1e-10 * max(1., float(np.max(np.abs(refvalues))))
    projector_error = float(np.linalg.norm(U[:, :2] @ U[:, :2].T - refU[:, :2] @ refU[:, :2].T, "fro")) if isolated else float("nan")
    return dict(
        residual=cert["actual_residual"], orth=cert["orthogonality_error"],
        reconstruction=float(np.linalg.norm(L - (U * values) @ U.T, "fro") / norm),
        evalerr=float(np.linalg.norm(values - refvalues) / max(np.linalg.norm(refvalues), 1e-30)),
        rank2projectorerr=projector_error, rank2_gap=gap,
        rank2_isolated=bool(isolated),
    )


def _mathematical_checks():
    """No signal Monte Carlo, formal timing study, or true-state inference."""
    started = time.perf_counter()
    rows = []
    periodic_failures = 0
    setup_info = None
    for clique_size in (3, 6):
        n = 2 * clique_size
        edges = []
        for offset in (0, clique_size):
            edges.extend((offset + i, offset + j) for i in range(clique_size) for j in range(i + 1, clique_size))
        edges.append((clique_size - 1, clique_size))
        B = np.zeros((n, len(edges)))
        for j, (a, b) in enumerate(edges):
            B[a, j], B[b, j] = 1., -1.
        pair = SpectrumPair(n, B)
        if setup_info is None:
            setup_info = pair.setup()
        sequence = (.6, .1, 0., 0., .2, .6)
        for t, bridge in enumerate(sequence):
            theta = np.ones(len(edges))
            theta[-1] = bridge
            L = make_laplacian(B, theta)
            outputs = pair.step(theta, L)
            # Reference calculated only after all candidates have returned.
            refvalues, refU = np.linalg.eigh(L)
            assert np.count_nonzero(np.abs(refvalues) < 1e-10) == (2 if bridge == 0 else 1)
            for method, output in outputs.items():
                v, U, meta = output["values"], output["vectors"], output["meta"]
                assert v.shape == (n,) and U.shape == (n, n)
                assert np.all(np.diff(v) >= -1e-12)
                metrics = numerical_metrics(L, v, U, refvalues, refU)
                actual_pass = metrics["residual"] <= RESIDUAL_TOL and metrics["orth"] <= ORTH_TOL
                assert meta["numerical_pass"] == actual_pass
                if not method.startswith("periodic"):
                    assert actual_pass, (method, n, t, metrics)
                    assert metrics["rank2_isolated"]
                    assert metrics["rank2projectorerr"] < 1e-4
                elif not actual_pass:
                    periodic_failures += 1
                assert np.isfinite(output["elapsed_s"]) and output["elapsed_s"] > 0
                rows.append(dict(n=n, t=t, bridge=bridge, method=method, metrics=metrics,
                                 numerical_pass=actual_pass, refresh_reason=meta["refresh_reason"]))
    assert periodic_failures > 0, "Periodic stale approximations must remain visibly uncertified"
    # All-isolated graph must also return a complete zero eigensystem; no
    # particular rank-two projector is identifiable for this all-repeated root.
    empty = SpectrumPair(n, B)
    zero_out = empty.step(np.zeros(len(edges)))
    for method, output in zero_out.items():
        assert output["meta"]["numerical_pass"]
        assert np.array_equal(output["values"], np.zeros(n))
    return dict(status="passed", kind="pure spectral mathematical checks, not a formal performance batch",
                setup=setup_info, rows=rows, records=len(rows),
                periodic_uncertified_outputs=periodic_failures,
                disconnected_and_reconnected=True, repeated_zero_root=True,
                all_isolated_graph=True, wall_s=time.perf_counter() - started)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        report = _mathematical_checks()
    destination = Path(__file__).with_name("CHECK_SPECTRUM.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))

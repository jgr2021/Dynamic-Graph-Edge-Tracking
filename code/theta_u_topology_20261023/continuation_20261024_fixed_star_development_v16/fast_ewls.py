"""Common-statistics implementations of the five frozen EWLS/RLS formulas.

Only implementation costs change: G/h are supplied once by structured_stats,
all methods use the same compiled cyclic coordinate descent, and no dense H
or repeated Gram is formed.  Ridge, alpha, forgetting, score floors, thresholds,
maximum sweeps, and relative KKT tolerance match the frozen V2 sources.

Keys global_adaptive/localized_adaptive correspond respectively to V2
innovation_adaptive_rls/localized_adaptive_rls.  Both gates remain heuristic;
no calibrated edge posterior, chi-square law, or familywise guarantee is claimed.
An actual application must charge shared preprocessing and report setup()
compilation separately.  This file executes no benchmark or formal campaign.

Public solver solve_box_qp is also available to every KF/MAP baseline.  Each
estimator exposes G/h/V/last_J/last_rhs/last_alpha and audit_state() for auditing.
Returned metadata stays JSON serializable rather than duplicating m-by-m arrays
at every update.  The full V matrix is retained to match the frozen recurrence.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


RIDGE_ABSOLUTE = 1e-8
RIDGE_RELATIVE = 1e-12
CD_MAX_SWEEPS = 5000
KKT_RELATIVE_TOLERANCE = 1e-8
FAMILY_ALPHA = 0.001
SCORE_FLOOR_ABSOLUTE = 1e-12
SCORE_FLOOR_RELATIVE = 1e-12
_SETUP_RECEIPT = None


@njit(cache=False)
def _kkt_kernel(gradient, theta):
    residual = 0.0
    for j in range(theta.size):
        if theta[j] <= 1e-12:
            violation = max(-gradient[j], 0.0)
        elif theta[j] >= 1.0 - 1e-12:
            violation = max(gradient[j], 0.0)
        else:
            violation = abs(gradient[j])
        residual = max(residual, violation)
    return residual


@njit(cache=False)
def _cd_kernel(J, c, initial, max_sweeps, relative_tolerance):
    theta = np.minimum(np.maximum(initial.copy(), 0.0), 1.0)
    gradient = J @ theta - c
    scale = max(1.0, np.max(np.abs(c)))
    tolerance = relative_tolerance * scale
    iterations = 0
    residual = _kkt_kernel(gradient, theta)
    while residual > tolerance and iterations < max_sweeps:
        for j in range(theta.size):
            new = min(max(theta[j] - gradient[j] / J[j, j], 0.0), 1.0)
            change = new - theta[j]
            if change != 0.0:
                theta[j] = new
                for i in range(theta.size):
                    gradient[i] += change * J[i, j]
        iterations += 1
        # Same explicit gradient refresh and actual KKT check as frozen V2.
        gradient = J @ theta - c
        residual = _kkt_kernel(gradient, theta)
    return theta, iterations, residual, tolerance, scale


def setup():
    """Compile the common solver once; return its separately measured cost.

    Python/NumPy/SciPy/Numba module import time is outside this duration and
    should also be reported as setup if whole-process latency is of interest.
    """
    global _SETUP_RECEIPT
    if _SETUP_RECEIPT is not None:
        return {**_SETUP_RECEIPT, "already_compiled": True}
    started = time.perf_counter()
    _cd_kernel(np.eye(2), np.array([0.2, 0.4]), np.full(2, 0.3),
               CD_MAX_SWEEPS, KKT_RELATIVE_TOLERANCE)
    _SETUP_RECEIPT = dict(numba_compile_seconds=time.perf_counter() - started,
                          solver="numba_cyclic_coordinate_descent_box_quadratic",
                          fastmath=False, cache=False,
                          max_sweeps=CD_MAX_SWEEPS,
                          relative_kkt_tolerance=KKT_RELATIVE_TOLERANCE,
                          compilation_included_in_steady_state=False,
                          module_import_time_included=False)
    return {**_SETUP_RECEIPT, "already_compiled": False}


def solve_box_qp(J, c, initial):
    """Minimize .5*x.T J*x-c.T*x on [0,1]^m with common frozen precision."""
    J = np.ascontiguousarray(J, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    initial = np.ascontiguousarray(initial, dtype=np.float64)
    if c.ndim != 1 or not len(c) or initial.shape != c.shape or J.shape != (len(c), len(c)):
        raise ValueError("Expected matching nonempty J[m,m], c[m], initial[m]")
    if not (np.all(np.isfinite(J)) and np.all(np.isfinite(c))
            and np.all(np.isfinite(initial))) or np.any(np.diag(J) <= 0):
        raise ValueError("Finite inputs and positive QP diagonal required")
    theta, iterations, residual, tolerance, scale = _cd_kernel(
        J, c, initial, CD_MAX_SWEEPS, KKT_RELATIVE_TOLERANCE)
    converged = bool(np.isfinite(residual) and residual <= tolerance)
    return theta, dict(optimizer="numba_cyclic_coordinate_descent_box_quadratic",
                       iterations=int(iterations), converged=converged, success=converged,
                       kkt_residual=float(residual),
                       kkt_relative_residual=float(residual / scale),
                       kkt_tolerance=float(tolerance),
                       objective=float(0.5 * theta @ J @ theta - c @ theta),
                       failure_reason=None if converged else "coordinate_descent_kkt_budget_exhausted")


def _regularized_information(G):
    symmetric = (G + G.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(symmetric)
    floor = max(RIDGE_ABSOLUTE, RIDGE_RELATIVE * max(1.0, float(eigenvalues[-1])))
    ridge = max(floor, floor - float(eigenvalues[0]))
    J = symmetric + ridge * np.eye(G.shape[0])
    return J, dict(information_min_eigenvalue=float(eigenvalues[0]),
                   information_max_eigenvalue=float(eigenvalues[-1]),
                   regularized_information_min_eigenvalue=float(eigenvalues[0] + ridge),
                   information_ridge=float(ridge))


def _predictive_nis(stats, theta, J_previous):
    # d = H.T innovation/sigma**2 = h-G theta.  The conventional formula is
    # cnu-d.T solve(J_previous+G,d), cnu=||r-H theta||**2/sigma**2.
    # It subtracts two potentially large terms.  Use the equivalent frozen
    # nonnegative variational expression instead, evaluating the remaining
    # residual through node states/Laplacian without forming H or redoing G.
    d = stats.h - stats.G @ theta
    A = J_previous + stats.G
    A = (A + A.T) * 0.5
    correction = cho_solve(cho_factor(A, lower=True, check_finite=False),
                           d, check_finite=False)
    residual_term = stats.residual_squared(theta + correction, stable=True) / stats.sigma**2
    prior_term = float(correction @ J_previous @ correction)
    value = float(residual_term + prior_term)
    guard = 128.0 * np.finfo(float).eps * max(1.0, abs(residual_term), abs(prior_term))
    # Only floating-point-scale negatives may be clipped.  A substantial
    # negative value is a visible numerical failure, never silently accepted.
    if not np.isfinite(value) or value < -max(1e-8, guard):
        raise FloatingPointError("Invalid predictive NIS")
    return max(0.0, value), dict(predictive_nis_computation="nonnegative_variational_observed_state_residual_no_H",
                               predictive_nis_roundoff_guard=float(guard),
                               predictive_nis_roundoff_clipped=bool(value < 0.0))


def _localized_scores(stats, theta, J_previous):
    G = stats.G
    d = stats.h - G @ theta
    A = J_previous + G
    A = (A + A.T) * 0.5
    factor = cho_factor(A, lower=True, check_finite=False)
    solved = cho_solve(factor, np.column_stack((d, G)), check_finite=False)
    u = d - G @ solved[:, 0]
    K = G - G @ solved[:, 1:]
    K = (K + K.T) * 0.5
    diagonal = np.diag(K)
    floor = max(SCORE_FLOOR_ABSOLUTE,
                SCORE_FLOOR_RELATIVE * max(1.0, float(np.max(np.diag(G)))))
    if float(np.min(diagonal)) < -100.0 * floor:
        raise FloatingPointError("Localized score variance has a substantive negative diagonal")
    z2 = u**2 / np.maximum(diagonal, floor)
    if np.any(~np.isfinite(z2)) or np.any(z2 < 0):
        raise FloatingPointError("Nonfinite localized score statistic")
    return u, K, z2, dict(localized_score_floor=float(floor),
                          localized_K_diagonal_min=float(np.min(diagonal)),
                          localized_K_diagonal_max=float(np.max(diagonal)),
                          localized_floor_used_coordinates=int(np.count_nonzero(diagonal < floor)))


class FastEWLS:
    def __init__(self, m, name, forgetting, gate="none"):
        if not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("m must be a positive integer")
        if gate not in ("none", "global", "localized"):
            raise ValueError("Unknown forgetting gate")
        self.m, self.name = int(m), str(name)
        self.forgetting, self.gate = float(forgetting), gate
        self.G, self.h, self.V = np.zeros((m, m)), np.zeros(m), np.zeros((m, m))
        self.theta, self.step = np.full(m, 0.3), 0
        self.local_threshold = float(chi2.ppf(1.0 - FAMILY_ALPHA / self.m, df=1))
        self.last_J, self.last_rhs, self.last_alpha = None, None, None

    def audit_state(self, copy=True):
        result = dict(G=self.G, h=self.h, V=self.V, theta=self.theta,
                      J=self.last_J, rhs=self.last_rhs, alpha=self.last_alpha,
                      step=self.step)
        return {key: value.copy() if copy and isinstance(value, np.ndarray) else value
                for key, value in result.items()}

    def update(self, stats):
        G, h = np.asarray(stats.G, dtype=float), np.asarray(stats.h, dtype=float)
        if G.shape != (self.m, self.m) or h.shape != (self.m,):
            raise ValueError("Common statistics do not match the estimator dimension")
        if not np.all(np.isfinite(G)) or not np.all(np.isfinite(h)) or stats.row_count <= 0:
            raise ValueError("Finite nonempty common statistics required")
        forgetting, triggered = self.forgetting, False
        nis = nis_threshold = local_threshold = max_z2 = None
        crossing, z2 = [], None
        gate_meta = dict(localized_score_floor=None, localized_K_diagonal_min=None,
                         localized_K_diagonal_max=None, localized_floor_used_coordinates=0)
        if self.gate != "none" and self.step:
            J_previous, _ = _regularized_information(self.G)
            if self.gate == "global":
                nis, nis_meta = _predictive_nis(stats, self.theta, J_previous)
                gate_meta.update(nis_meta)
                nis_threshold = float(chi2.ppf(0.999, stats.row_count))
                triggered = bool(nis > nis_threshold)
            else:
                _, _, z2, score_meta = _localized_scores(stats, self.theta, J_previous)
                gate_meta.update(score_meta)
                local_threshold = self.local_threshold
                max_z2 = float(np.max(z2))
                crossing = np.flatnonzero(z2 > local_threshold).tolist()
                triggered = bool(crossing)
            forgetting = 0.20 if triggered else 0.97
        self.G = forgetting * self.G + G
        self.h = forgetting * self.h + h
        self.V = forgetting**2 * self.V + G
        alpha = 0.5 * np.sqrt(2.0 * np.log(max(2, self.m)) * np.maximum(0.0, np.diag(self.V)))
        J, information_meta = _regularized_information(self.G)
        rhs = self.h - alpha
        self.theta, meta = solve_box_qp(J, rhs, self.theta)
        self.step += 1
        self.last_J, self.last_rhs, self.last_alpha = J, rhs, alpha
        meta.update(information_meta)
        meta.update(gate_meta)
        meta.update(method=self.name, step=self.step, forgetting=float(forgetting),
                    frozen_formula_method={"global_adaptive": "innovation_adaptive_rls",
                                           "localized_adaptive": "localized_adaptive_rls"}.get(self.name, self.name),
                    l1_alpha_min=float(np.min(alpha)), l1_alpha_max=float(np.max(alpha)),
                    triggered=triggered, predictive_nis=nis, predictive_nis_threshold=nis_threshold,
                    predictive_nis_df=int(stats.row_count) if self.gate == "global" else None,
                    predictive_nis_calibration="heuristic_constrained_l1_information_proxy"
                        if self.gate == "global" else "not_applicable",
                    localized_max_z2=max_z2, localized_threshold=local_threshold,
                    localized_nominal_family_alpha=FAMILY_ALPHA if self.gate == "localized" else None,
                    localized_coordinate_count=self.m if self.gate == "localized" else None,
                    localized_triggered_edge_indices=crossing,
                    localized_score_z2=None if z2 is None else z2.tolist(),
                    localized_calibration="heuristic_only_no_exact_chisquare_or_familywise_claim"
                        if self.gate == "localized" else "not_applicable",
                    block_rows=int(stats.row_count),
                    block_residual_squared=stats.residual_squared(self.theta, stable=True),
                    posterior_covariance_claimed=False, shared_observation_statistics=True,
                    gram_recomputed_in_update=False, common_qp_solver=True)
        return self.theta.copy(), meta


def make_ewls(m):
    return {"block_nnls": FastEWLS(m, "block_nnls", 0.0),
            "ewls_fast": FastEWLS(m, "ewls_fast", 0.70),
            "ewls_slow": FastEWLS(m, "ewls_slow", 0.97),
            "global_adaptive": FastEWLS(m, "global_adaptive", 0.97, gate="global"),
            "localized_adaptive": FastEWLS(m, "localized_adaptive", 0.97, gate="localized")}


def _read_frozen_modules():
    """Load source in memory only; never create a frozen-directory pycache."""
    directory = Path(__file__).resolve().parent.parent / "topology_v2"
    base = types.ModuleType("_frozen_ewls_base")
    base.__file__ = str(directory / "filters.py")
    exec(compile((directory / "filters.py").read_text(encoding="utf-8"), base.__file__, "exec"), base.__dict__)
    previous = sys.modules.get("filters")
    sys.modules["filters"] = base
    try:
        local = types.ModuleType("_frozen_ewls_localized")
        local.__file__ = str(directory / "localized_filter.py")
        exec(compile((directory / "localized_filter.py").read_text(encoding="utf-8"), local.__file__, "exec"), local.__dict__)
    finally:
        if previous is None:
            del sys.modules["filters"]
        else:
            sys.modules["filters"] = previous
    return base, local


def self_check():
    """Deterministic same-H/r multiblock comparisons, not benefit evaluation."""
    from structured_stats import EdgeObservationStructure

    directory = Path(__file__).resolve().parent
    protected = [directory.parent / "topology_v2" / name
                 for name in ("filters.py", "localized_filter.py", "model.py")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    compile_receipt = setup()
    old, old_local = _read_frozen_modules()
    rng = np.random.default_rng(9194735)
    records = []
    A = rng.normal(size=(9, 9))
    J, c, initial = A.T @ A + 0.2 * np.eye(9), rng.normal(size=9), rng.uniform(-0.1, 1.1, 9)
    actual, meta = solve_box_qp(J, c, initial)
    expected, expected_meta = old._box_quadratic(J, c, initial)
    assert np.allclose(actual, expected, atol=2e-10, rtol=2e-10)
    assert meta["success"] and expected_meta["success"]
    records.append(dict(case="common_compiled_solver_matches_frozen_python", max_theta_abs_error=float(np.max(np.abs(actual-expected)))))

    mapping = {"global_adaptive": "innovation_adaptive_rls", "localized_adaptive": "localized_adaptive_rls"}
    for case, n, M, blocks in (("excited_changing_weights", 6, 12, 5),
                                ("consensus_zero_information", 6, 8, 3),
                                ("large_common_modes", 4, 9, 3)):
        edges = np.asarray([(i, j) for i in range(n) for j in range(i+1, n)], dtype=int)
        B = np.zeros((n, len(edges)))
        B[edges[:, 0], np.arange(len(edges))] = 1.0
        B[edges[:, 1], np.arange(len(edges))] = -1.0
        structure = EdgeObservationStructure(n, edges)
        fast, frozen = make_ewls(len(edges)), old_local.make_filters6(len(edges))
        maxima = dict(theta=0.0, G=0.0, h=0.0, V=0.0, rhs=0.0, objective=0.0, nis=0.0, z2=0.0)
        max_nis_relative_error = 0.0
        trigger_records = []
        for step in range(blocks):
            states = rng.normal(size=(M, n))
            if case == "consensus_zero_information":
                states = np.repeat(rng.normal(size=(M, 1)), n, axis=1)
            if case == "large_common_modes":
                states += 1e6 * rng.normal(size=(M, 1))
            H = np.vstack([(.8/n) * B * (B.T @ s)[None, :] for s in states])
            truth = np.full(len(edges), 0.3)
            truth[0] = 0.85 if step >= 2 else 0.2
            r = H @ truth + rng.normal(0, 0.03, M*n)
            if case == "large_common_modes":
                r = (r.reshape(M, n) + 1e5 * rng.normal(size=(M, 1))).reshape(-1)
            block = structure.statistics(states, r.reshape(M, n), .8/n, .03)
            for key, method in fast.items():
                reference = frozen[mapping.get(key, key)]
                theta, fast_meta = method.update(block)
                old_theta, old_meta = reference.update(H, r, .03)
                assert fast_meta["success"] and old_meta["success"], (case, key, step)
                assert fast_meta["triggered"] == old_meta["triggered"], (case, key, step)
                assert fast_meta["forgetting"] == old_meta["forgetting"], (case, key, step)
                assert fast_meta["localized_triggered_edge_indices"] == old_meta.get("localized_triggered_edge_indices", []), (case, key, step)
                error = float(np.max(np.abs(theta-old_theta)))
                assert error <= 2e-6, (case, key, step, error)
                maxima["theta"] = max(maxima["theta"], error)
                for attr in ("G", "h", "V"):
                    value, expected = getattr(method, attr), getattr(reference, attr)
                    assert np.allclose(value, expected, atol=2e-5, rtol=2e-8), (case, key, step, attr)
                    maxima[attr] = max(maxima[attr], float(np.max(np.abs(value-expected))))
                old_J, _ = old._regularized_information(reference.G)
                old_alpha = .5*np.sqrt(2*np.log(max(2,len(edges)))*np.maximum(0,np.diag(reference.V)))
                old_rhs = reference.h-old_alpha
                assert np.allclose(method.last_rhs, old_rhs, atol=2e-5, rtol=2e-8)
                maxima["rhs"] = max(maxima["rhs"], float(np.max(np.abs(method.last_rhs-old_rhs))))
                # Independently recheck the actual KKT condition in the fast QP.
                actual_kkt = old._kkt_residual(method.last_J, method.last_rhs, theta)
                assert actual_kkt <= fast_meta["kkt_tolerance"] * (1+1e-7) + 1e-10
                objective_old_qp = float(.5*theta@old_J@theta-old_rhs@theta)
                maxima["objective"] = max(maxima["objective"], abs(objective_old_qp-old_meta["objective"]))
                if fast_meta["predictive_nis"] is not None:
                    value, expected = fast_meta["predictive_nis"], old_meta["predictive_nis"]
                    assert np.isclose(value, expected, atol=2e-7, rtol=2e-8)
                    maxima["nis"] = max(maxima["nis"], abs(value-expected))
                    max_nis_relative_error = max(max_nis_relative_error, abs(value-expected)/max(1.0,abs(expected)))
                if fast_meta["localized_score_z2"] is not None:
                    value, expected = np.asarray(fast_meta["localized_score_z2"]), np.asarray(old_meta["localized_score_z2"])
                    assert np.allclose(value, expected, atol=2e-5, rtol=2e-7)
                    maxima["z2"] = max(maxima["z2"], float(np.max(np.abs(value-expected))))
                trigger_records.append(dict(step=step, method=key, triggered=fast_meta["triggered"],
                                            forgetting=fast_meta["forgetting"], theta_max_abs_error=error,
                                            kkt_relative_residual=fast_meta["kkt_relative_residual"]))
                assert not fast_meta["posterior_covariance_claimed"] and not fast_meta["gram_recomputed_in_update"]
        records.append(dict(case=case, n=n, M=M, blocks=blocks, method_updates=blocks*5,
                            max_abs_errors=maxima, max_predictive_nis_relative_error=max_nis_relative_error,
                            all_triggers_and_forgetting_match=True,
                            all_actual_KKT_checks_pass=True, update_records=trigger_records))
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    assert before == after
    return dict(status="passed", scope="deterministic_algebra_equivalence_only_no_benchmark_no_tuning",
                diagnostic_rng_seed=9194735, compilation=compile_receipt, checks=records,
                frozen_source_sha256=after, frozen_sources_unchanged=True,
                new_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    result = self_check()
    Path(__file__).with_name("CHECK_FAST_EWLS.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "checks": [{key: value for key, value in row.items() if key != "update_records"}
                                 for row in result["checks"]]}, indent=2, ensure_ascii=False))

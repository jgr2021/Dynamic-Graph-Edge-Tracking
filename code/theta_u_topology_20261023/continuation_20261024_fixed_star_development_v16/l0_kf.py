"""Fixed-penalty KF-L0 mechanism controls; no physical positive-weight gap.

make_l0_kf(m) returns kf_l0_logm and kf_l0_halflogm.  Both consume only the
common BlockStatistics, use Q=.02**2 I once per slow block, P0=.25**2 I and
mean .3, then approximately minimize .5*x'J*x-b'x+kappa*||x||0 on [0,1]^m.
This is an existing best-subset principle specialized to graph edge regression,
NOT a novelty claim or a calibrated support test.  Nonconvex solutions are
certified only for single-coordinate stability and active-support box KKT,
not global optimality.  See LITERATURE_BOUNDARY_ZH.md for primary sources.

Two deterministic starts are clip(Gaussian mean) and all zeros, in that order.
Each of at most 30 outer rounds performs an exact full-coordinate L0 sweep,
then uses the SAME fast_ewls.solve_box_qp on the selected support without L1
penalty.  Its inner 5000-sweep budget is unchanged.  A full-candidate exact
coordinate certificate follows every refit: inactive candidates are never
excluded from the scan.  The lower actual objective wins; an exact tie picks
the Gaussian-start branch.  All branch/inner-solver failures are retained and
make overall success False, even if the other branch is stable.  The returned
best actual iterate is carried into the next block; no failed iterate is
silently reset to a previous state or restarted with a larger budget.

Full Gaussian P=J^-1 is carried without support masking, allowing removed
edges to reactivate.  It is NOT the sparse/constrained posterior covariance.
Default online guards use necessary prior/J Cholesky, finite symmetric P with
positive diagonal, and O(m**2) mean backward error.  Full P Cholesky and J@P
verification appear only in the independent diagnostics below.

Call setup() and fast_ewls.setup() once before steady timing and report their
compilation separately.  Running --check writes only CHECK_L0_KF.json; there
is no formal-experiment entry point and no truth is accepted by the estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve

from fast_ewls import solve_box_qp
from matched_kf import PROCESS_VARIANCE, INITIAL_VARIANCE, INITIAL_MEAN
from structured_stats import BlockStatistics


MAX_OUTER_ROUNDS = 30
COORDINATE_RELATIVE_TOLERANCE = 1e-10
KKT_RELATIVE_TOLERANCE = 1e-8
MONOTONIC_RELATIVE_TOLERANCE = 1e-12
_SETUP_RECEIPT = None


@njit(cache=False)
def _coordinate_sweep(J, b, kappa, initial):
    x = initial.copy()
    gradient = J @ x - b
    changed = 0
    largest_conditional_increase = 0.0
    for e in range(x.size):
        diagonal = J[e, e]
        d = diagonal * x[e] - gradient[e]
        u = min(max(d / diagonal, 0.0), 1.0)
        gain = d * u - 0.5 * diagonal * u * u
        new = u if gain > kappa else 0.0  # exact equality chooses zero
        old = x[e]
        old_cost = 0.5 * diagonal * old * old - d * old + (kappa if old != 0.0 else 0.0)
        new_cost = 0.5 * diagonal * new * new - d * new + (kappa if new != 0.0 else 0.0)
        largest_conditional_increase = max(largest_conditional_increase, new_cost - old_cost)
        delta = new - old
        if delta != 0.0:
            changed += 1
            x[e] = new
            for f in range(x.size):
                gradient[f] += delta * J[f, e]
    return x, changed, largest_conditional_increase


@njit(cache=False)
def _coordinate_certificate(J, b, kappa, x):
    gradient = J @ x - b
    maximum_gap = 0.0
    for e in range(x.size):
        diagonal = J[e, e]
        d = diagonal * x[e] - gradient[e]
        u = min(max(d / diagonal, 0.0), 1.0)
        current = 0.5 * diagonal * x[e] * x[e] - d * x[e] + (kappa if x[e] != 0.0 else 0.0)
        candidate = 0.5 * diagonal * u * u - d * u + (kappa if u != 0.0 else 0.0)
        maximum_gap = max(maximum_gap, current - min(0.0, candidate))
    return max(0.0, maximum_gap)


def setup():
    """Compile only L0 kernels, using no scientific observation stream."""
    global _SETUP_RECEIPT
    if _SETUP_RECEIPT is not None:
        return {**_SETUP_RECEIPT, "already_compiled": True}
    started = time.perf_counter()
    J = np.array([[2., .2], [.2, 3.]])
    b = np.array([.3, 1.4])
    x, _, _ = _coordinate_sweep(J, b, .4, np.array([.2, .2]))
    _coordinate_certificate(J, b, .4, x)
    _SETUP_RECEIPT = dict(numba_compile_seconds=time.perf_counter() - started,
                          max_outer_rounds=MAX_OUTER_ROUNDS, fastmath=False,
                          cache=False, coordinate_relative_tolerance=COORDINATE_RELATIVE_TOLERANCE,
                          support_kkt_relative_tolerance=KKT_RELATIVE_TOLERANCE,
                          compilation_included_in_steady_state=False,
                          common_box_qp_setup_separate=True)
    return {**_SETUP_RECEIPT, "already_compiled": False}


def objective(J, b, kappa, x):
    return float(.5 * x @ J @ x - b @ x + kappa * np.count_nonzero(x))


def _active_kkt(J, b, x):
    active = np.flatnonzero(x > 0.0)
    if not len(active):
        return 0.0
    g = (J @ x - b)[active]
    violation = np.abs(g)
    upper = x[active] >= 1.0 - 1e-12
    violation[upper] = np.maximum(g[upper], 0.0)
    return float(np.max(violation))


def _solve_start(J, b, kappa, initial, label):
    x = np.clip(np.asarray(initial, dtype=float), 0., 1.).copy()
    current_objective = objective(J, b, kappa, x)
    history = [current_objective]
    qp_iterations = qp_calls = changed_coordinates = 0
    failure_reasons, inner_failures = [], []
    stable = False
    gap = float("inf")
    kkt = float("inf")
    rounds = 0
    for outer in range(MAX_OUTER_ROUNDS):
        rounds = outer + 1
        x, changed, conditional_increase = _coordinate_sweep(J, b, kappa, x)
        changed_coordinates += int(changed)
        swept_objective = objective(J, b, kappa, x)
        monotonic_scale = max(1., abs(current_objective), abs(swept_objective))
        if conditional_increase > MONOTONIC_RELATIVE_TOLERANCE * monotonic_scale \
                or swept_objective > current_objective + MONOTONIC_RELATIVE_TOLERANCE * monotonic_scale:
            failure_reasons.append("coordinate_objective_increase")
        history.append(swept_objective)
        active = np.flatnonzero(x > 0.0)
        if len(active):
            fitted, qp_meta = solve_box_qp(J[np.ix_(active, active)], b[active], x[active])
            x = np.zeros_like(x)
            x[active] = fitted
            qp_iterations += int(qp_meta["iterations"])
            qp_calls += 1
            if not qp_meta["success"]:
                inner_failures.append(dict(round=rounds, **qp_meta))
                failure_reasons.append("support_qp_failed")
        fitted_objective = objective(J, b, kappa, x)
        monotonic_scale = max(1., abs(swept_objective), abs(fitted_objective))
        if fitted_objective > swept_objective + MONOTONIC_RELATIVE_TOLERANCE * monotonic_scale:
            failure_reasons.append("support_refit_objective_increase")
        history.append(fitted_objective)
        current_objective = fitted_objective
        if not np.all(np.isfinite(x)) or not np.isfinite(current_objective) or np.any((x < 0) | (x > 1)):
            raise FloatingPointError("Invalid L0 coordinate/refit output")
        gap = float(_coordinate_certificate(J, b, kappa, x))
        kkt = _active_kkt(J, b, x)
        objective_scale = max(1., abs(current_objective), kappa)
        gradient_scale = max(1., float(np.max(np.abs(b))))
        stable = bool(gap <= COORDINATE_RELATIVE_TOLERANCE * objective_scale
                      and kkt <= KKT_RELATIVE_TOLERANCE * gradient_scale)
        if failure_reasons or stable:
            break
    if not stable and not failure_reasons:
        failure_reasons.append("l0_outer_budget_exhausted")
    return x, dict(start=label, objective=current_objective, objective_history=history,
                   outer_rounds=rounds, coordinate_sweeps=rounds,
                   changed_coordinates=changed_coordinates, support_qp_calls=qp_calls,
                   support_qp_iterations=qp_iterations, inner_failures=inner_failures,
                   coordinate_gap=gap,
                   coordinate_relative_gap=gap / max(1., abs(current_objective), kappa),
                   active_kkt_residual=kkt,
                   active_kkt_relative_residual=kkt / max(1., float(np.max(np.abs(b)))),
                   coordinate_stable=stable, success=bool(stable and not failure_reasons),
                   failure_reasons=failure_reasons, support_size=int(np.count_nonzero(x)))


def solve_l0_box(J, b, kappa, gaussian_mean):
    """Two fixed starts; best real objective, with exact ties favoring start 0.

    This solver returns coordinate certificates only.  It does not enumerate
    supports or use truth, sparsity cardinality, physical gap, or event times.
    """
    J, b = np.asarray(J, dtype=float), np.asarray(b, dtype=float)
    gaussian_mean = np.asarray(gaussian_mean, dtype=float)
    if b.ndim != 1 or not len(b) or J.shape != (len(b), len(b)) or gaussian_mean.shape != b.shape:
        raise ValueError("Expected matching J[m,m], b[m], Gaussian mean[m]")
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(gaussian_mean)):
        raise ValueError("Finite L0 inputs required")
    if not np.isfinite(kappa) or kappa <= 0 or np.any(np.diag(J) <= 0):
        raise ValueError("Positive finite kappa and information diagonal required")
    asymmetry = np.max(np.abs(J - J.T))
    if asymmetry > 1e-10 * max(1., float(np.max(np.abs(J)))):
        raise ValueError("Symmetric information required")
    starts = [np.clip(gaussian_mean, 0., 1.), np.zeros_like(b)]
    labels = ["clipped_gaussian", "all_zero"]
    outputs = [_solve_start(J, b, kappa, initial, label) for initial, label in zip(starts, labels)]
    chosen = 0 if outputs[0][1]["objective"] <= outputs[1][1]["objective"] else 1
    x, selected = outputs[chosen]
    starts_meta = [item[1] for item in outputs]
    success = all(item["success"] for item in starts_meta)
    reasons = [item["start"] + ":" + reason for item in starts_meta for reason in item["failure_reasons"]]
    return x, dict(optimizer="two_start_exact_l0_coordinate_and_common_support_box_qp",
                   iterations=sum(item["coordinate_sweeps"] + item["support_qp_iterations"] for item in starts_meta),
                   outer_iterations_total=sum(item["outer_rounds"] for item in starts_meta),
                   support_qp_calls=sum(item["support_qp_calls"] for item in starts_meta),
                   support_qp_iterations=sum(item["support_qp_iterations"] for item in starts_meta),
                   converged=success, success=success,
                   failure_reason=";".join(reasons) if reasons else None,
                   selected_start_index=chosen, selected_start=labels[chosen],
                   tie_rule="exact_objective_tie_prefers_clipped_gaussian_start",
                   objective=selected["objective"], coordinate_stable=selected["coordinate_stable"],
                   coordinate_certificate_gap=selected["coordinate_gap"],
                   coordinate_relative_gap=selected["coordinate_relative_gap"],
                   coordinate_tolerance=COORDINATE_RELATIVE_TOLERANCE * max(1., abs(selected["objective"]), kappa),
                   kkt_residual=selected["active_kkt_residual"],
                   kkt_relative_residual=selected["active_kkt_relative_residual"],
                   kkt_tolerance=KKT_RELATIVE_TOLERANCE * max(1., float(np.max(np.abs(b)))),
                   kkt_target="active_support_box_quadratic_plus_full_l0_coordinate_certificate",
                   global_optimality_claimed=False, starts=starts_meta)


class L0KF:
    def __init__(self, m, name, multiplier):
        if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
            raise ValueError("m must be a positive integer")
        self.m, self.name = int(m), str(name)
        self.kappa = float(multiplier) * float(np.log(max(2, self.m)))
        self.theta, self.step = np.full(self.m, INITIAL_MEAN), 0
        self.P, self.Q = INITIAL_VARIANCE * np.eye(self.m), PROCESS_VARIANCE * np.eye(self.m)
        self.last_prior_mean = self.last_prior_P = None
        self.last_J = self.last_rhs = self.last_alpha = None
        self.last_gaussian_mean = self.last_G = self.last_h = None

    def audit_state(self, copy=True):
        result = dict(theta=self.theta, P=self.P, Q=self.Q, step=self.step, kappa=self.kappa,
                      prior_mean=self.last_prior_mean, prior_P=self.last_prior_P,
                      J=self.last_J, rhs=self.last_rhs, alpha=self.last_alpha,
                      gaussian_mean=self.last_gaussian_mean, G=self.last_G, h=self.last_h,
                      support=self.theta > 0)
        return {key: value.copy() if copy and isinstance(value, np.ndarray) else value
                for key, value in result.items()}

    def update(self, stats: BlockStatistics):
        if not isinstance(stats, BlockStatistics):
            raise TypeError("Expected common structured_stats.BlockStatistics")
        G, h = np.asarray(stats.G, dtype=float), np.asarray(stats.h, dtype=float)
        if G.shape != (self.m, self.m) or h.shape != (self.m,) or stats.row_count <= 0:
            raise ValueError("Common statistics dimension/row count mismatch")
        if not np.all(np.isfinite(G)) or not np.all(np.isfinite(h)):
            raise ValueError("Finite common statistics required")
        if np.max(np.abs(G - G.T)) > 1e-10 * max(1., float(np.max(np.abs(G)))):
            raise ValueError("Symmetric observation information required")
        G = (G + G.T) * .5
        prior_mean, prior_P = self.theta.copy(), self.P + self.Q
        identity = np.eye(self.m)
        prior_precision = cho_solve(cho_factor(prior_P, lower=True), identity, check_finite=False)
        J = prior_precision + G
        J = (J + J.T) * .5
        b = prior_precision @ prior_mean + h
        solved = cho_solve(cho_factor(J, lower=True), np.column_stack((b, identity)), check_finite=False)
        gaussian = solved[:, 0]
        P = (solved[:, 1:] + solved[:, 1:].T) * .5
        if not np.all(np.isfinite(P)) or np.any(np.diag(P) <= 0):
            raise np.linalg.LinAlgError("Nonfinite Gaussian covariance or nonpositive diagonal")
        mean_residual = float(np.linalg.norm(J @ gaussian - b, np.inf))
        backward_scale = max(1., float(np.linalg.norm(J, np.inf) * np.linalg.norm(gaussian, np.inf)
                                      + np.linalg.norm(b, np.inf)))
        theta, meta = solve_l0_box(J, b, self.kappa, gaussian)
        # Preserve the chosen actual iterate even when any branch failed.
        # Covariance stays full; no support-masking or previous-state reset.
        self.theta, self.P = theta.copy(), P
        self.step += 1
        self.last_prior_mean, self.last_prior_P = prior_mean, prior_P
        self.last_J, self.last_rhs = J, b
        self.last_alpha = np.zeros(self.m)  # no L1 penalty; kappa is separate
        self.last_gaussian_mean, self.last_G, self.last_h = gaussian.copy(), G.copy(), h.copy()
        meta.update(method=self.name, step=self.step, kappa=self.kappa,
                    kappa_policy="log(max(2,m))" if self.name == "kf_l0_logm" else ".5*log(max(2,m))",
                    fixed_start_order=["clipped_gaussian", "all_zero"],
                    max_outer_rounds_per_start=MAX_OUTER_ROUNDS, no_physical_gap_prior=True,
                    exact_zero_edges=int(np.count_nonzero(self.theta == 0)),
                    linear_solve_residual=mean_residual, gaussian_backward_error=mean_residual / backward_scale,
                    prior_covariance_spd=True, information_spd=True,
                    covariance_spd=None, covariance_full_cholesky_checked=False,
                    covariance_spd_interpretation="by_construction_information_inverse_not_full_P_chol_checked",
                    covariance_diagonal_min=float(np.min(np.diag(P))),
                    covariance_symmetry_error=float(np.max(np.abs(P - P.T))),
                    posterior_covariance_claimed=False,
                    covariance_interpretation="full_unconstrained_gaussian_carried_after_l0_point_update",
                    covariance_accounts_for_support_selection=False, support_rows_permanently_masked=False,
                    process_variance_per_slow_block=PROCESS_VARIANCE,
                    process_covariance_injections_this_block=1,
                    block_rows=int(stats.row_count),
                    block_residual_squared=stats.residual_squared(self.theta, stable=True),
                    triggered=False, shared_observation_statistics=True, gram_recomputed_in_update=False,
                    common_qp_solver=True, no_truth_input=True)
        return self.theta.copy(), meta


def make_l0_kf(m):
    return {"kf_l0_logm": L0KF(m, "kf_l0_logm", 1.),
            "kf_l0_halflogm": L0KF(m, "kf_l0_halflogm", .5)}


def run_diagnostics():
    """Small algebra fixtures, not research streams; diagnostic seeds 9161xx."""
    import scipy
    from scipy.linalg import cholesky, solve_triangular
    from scipy.optimize import lsq_linear
    import fast_ewls
    from structured_stats import EdgeObservationStructure

    setup_receipt = dict(l0=setup(), common_qp=fast_ewls.setup())
    checks = []
    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))

    def independent_support_solution(J, b, mask):
        ids = np.flatnonzero(mask)
        x = np.zeros_like(b)
        if len(ids):
            sub = J[np.ix_(ids, ids)]
            C = cholesky(sub, lower=True).T
            y = solve_triangular(C.T, b[ids], lower=True)
            fit = lsq_linear(C, y, bounds=(0., 1.), method="bvls", tol=1e-13, max_iter=1000)
            if not fit.success:
                raise AssertionError("Independent SciPy support fit failed")
            x[ids] = fit.x
        return x

    def enumerate_global(J, b, kappa):
        best_x, best_f = np.zeros_like(b), float("inf")
        for mask in itertools.product((False, True), repeat=len(b)):
            x = independent_support_solution(J, b, mask)
            value = objective(J, b, kappa, x)
            if value < best_f:
                best_x, best_f = x, value
        return best_x, best_f

    manual_J = np.diag([4., 9., 16.])
    manual_b, manual_kappa = np.array([.2, 12., 2.]), .4
    hand_expected = np.array([0., 1., 0.])
    hand_x, hand_meta = solve_l0_box(manual_J, manual_b, manual_kappa,
                                    np.linalg.solve(manual_J, manual_b))
    global_x, global_f = enumerate_global(manual_J, manual_b, manual_kappa)
    check("manual_zero_vs_unshrunk_upper_bound_and_enumeration",
          np.allclose(hand_x, hand_expected, atol=1e-12, rtol=0)
          and abs(hand_meta["objective"] - global_f) < 1e-12 and hand_meta["success"],
          theta=hand_x.tolist(), objective=hand_meta["objective"], enumerated_objective=global_f)

    # At exact zero-vs-positive equality, the coordinate rule chooses zero.
    tie_x, _, _ = _coordinate_sweep(np.array([[2.]]), np.array([1.]), .25, np.array([.5]))
    check("exact_coordinate_tie_chooses_zero", tie_x[0] == 0.)
    same_x, same_meta = solve_l0_box(np.eye(2), np.zeros(2), .5, np.zeros(2))
    check("exact_two_start_objective_tie_prefers_gaussian_start",
          np.array_equal(same_x, np.zeros(2)) and same_meta["selected_start_index"] == 0)

    examples = [
        (np.array([[4., 3.2], [3.2, 4.]]), np.array([3., 3.]), .4, "correlated_redundant_edges"),
        (np.array([[2., -1.8], [-1.8, 2.]]), np.array([.3, .3]), .12, "correlated_synergistic_edges"),
        (np.diag([2., 3., 5.]), np.array([-1., 6., .2]), .2, "nonnegative_and_upper_bounds"),
    ]
    for seed, m in ((916101, 3), (916102, 4), (916103, 5), (916104, 6)):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(m + 3, m))
        J = A.T @ A + .3 * np.eye(m)
        b = J @ rng.uniform(-.2, 1.3, m)
        examples.append((J, b, .6, "seed_" + str(seed)))
    enumeration_cases = []
    for J, b, kappa, label in examples:
        x, meta = solve_l0_box(J, b, kappa, np.linalg.solve(J, b))
        _, global_f = enumerate_global(J, b, kappa)
        fitted = independent_support_solution(J, b, x > 0)
        support_error = float(np.max(np.abs(fitted - x)))
        local_gap = float(_coordinate_certificate(J, b, kappa, x))
        monotonic = all(all(after <= before + 1e-10 * max(1., abs(before), abs(after))
                           for before, after in zip(start["objective_history"][:-1], start["objective_history"][1:]))
                        for start in meta["starts"])
        record = dict(case=label, m=len(b), objective=meta["objective"],
                      exhaustive_support_count=2**len(b), exhaustive_global_objective=global_f,
                      nonnegative_global_objective_gap=meta["objective"] - global_f,
                      scipy_selected_support_theta_error=support_error,
                      coordinate_certificate_gap=local_gap, active_kkt_residual=meta["kkt_residual"],
                      monotonic=monotonic, selected_start=meta["selected_start"], success=meta["success"])
        enumeration_cases.append(record)
        check("independent_enumeration_local_certificate_" + label,
              meta["success"] and meta["global_optimality_claimed"] is False
              and meta["objective"] >= global_f - 1e-9
              and support_error < 2e-7 and local_gap <= meta["coordinate_tolerance"]
              and meta["kkt_residual"] <= meta["kkt_tolerance"] and monotonic,
              **record)

    # Actual shared-statistics API, including informative zero-edge evidence,
    # subsequent restoration and completely uninformative consensus blocks.
    n = 4
    edges = np.array(list(itertools.combinations(range(n), 2)), dtype=np.int64)
    structure = EdgeObservationStructure(n, edges)
    m = len(edges)
    rng = np.random.default_rng(916111)
    states = rng.normal(size=(48, n))
    eta, sigma = .8 / n, .03
    zero_stats = structure.statistics(states, np.zeros_like(states), eta, sigma)
    recovery_s = rng.normal(size=(48, n))
    recovery_r = eta * recovery_s @ structure.laplacian(np.full(m, .8))
    recovery_stats = structure.statistics(recovery_s, recovery_r, eta, sigma)
    consensus = np.repeat(rng.normal(size=(48, 1)), n, axis=1)
    noinfo_stats = structure.statistics(consensus, np.ones_like(consensus), eta, sigma)
    api_cases = []
    for name, estimator in make_l0_kf(m).items():
        deleted, deleted_meta = estimator.update(zero_stats)
        deletion_state = estimator.audit_state()
        restored, restored_meta = estimator.update(recovery_stats)
        restored_state = estimator.audit_state()
        before_P = estimator.P.copy()
        _, noinfo_meta = estimator.update(noinfo_stats)
        after_state = estimator.audit_state()
        inverse_error = float(np.linalg.norm(after_state["J"] @ after_state["P"] - np.eye(m), np.inf))
        full_chol_minimum = float(np.min(np.diag(cholesky(after_state["P"], lower=True))))
        gaussian_ref = np.linalg.solve(restored_state["J"], restored_state["rhs"])
        case = dict(method=name, removed_edges=int(np.count_nonzero(deleted == 0)),
                    restored_edges=int(np.count_nonzero(restored > 0)),
                    restored_gaussian_error=float(np.max(np.abs(gaussian_ref - restored_state["gaussian_mean"]))),
                    noinfo_single_Q_error=float(np.max(np.abs(estimator.P - before_P - PROCESS_VARIANCE * np.eye(m)))),
                    independent_inverse_error=inverse_error,
                    independent_full_P_cholesky_min_diagonal=full_chol_minimum,
                    min_deleted_edge_variance=float(np.min(np.diag(deletion_state["P"]))))
        api_cases.append(case)
        check("shared_stats_full_covariance_delete_restore_noinfo_" + name,
              np.all(deleted == 0) and np.all(restored > 0)
              and deleted_meta["success"] and restored_meta["success"] and noinfo_meta["success"]
              and case["restored_gaussian_error"] < 1e-11
              and case["noinfo_single_Q_error"] < 1e-12
              and inverse_error < 1e-11 and full_chol_minimum > 0
              and case["min_deleted_edge_variance"] > 0,
              **case)
        copied = estimator.audit_state()
        protected = estimator.P.copy()
        copied["P"].fill(-1.)
        copied["J"].fill(-1.)
        restored.fill(-1.)
        check("audit_and_output_copy_no_matrix_JSON_" + name,
              np.array_equal(estimator.P, protected) and np.all(estimator.theta >= 0)
              and not any(isinstance(item, np.ndarray) for item in restored_meta.values())
              and restored_meta["posterior_covariance_claimed"] is False
              and restored_meta["support_rows_permanently_masked"] is False)
        json.dumps(restored_meta, allow_nan=False)

    # Inject only a callable failure into a finite synthetic SPD problem;
    # this checks bookkeeping, not an experiment or a changed solver budget.
    actual_solver = globals()["solve_box_qp"]
    def failed_solver(J, b, initial):
        return np.clip(initial, 0, 1), dict(iterations=5000, success=False,
            failure_reason="diagnostic_injected_budget_failure", converged=False,
            optimizer="diagnostic_only", kkt_residual=1., kkt_relative_residual=1.,
            kkt_tolerance=1e-8, objective=float(.5 * initial @ J @ initial - b @ initial))
    try:
        globals()["solve_box_qp"] = failed_solver
        failed_estimator = L0KF(m, "kf_l0_logm", 1.)
        failed_theta, failed_meta = failed_estimator.update(recovery_stats)
        check("inner_failure_retained_actual_iterate_not_reset",
              failed_meta["success"] is False and failed_meta["converged"] is False
              and failed_meta["failure_reason"] is not None
              and any(start["inner_failures"] for start in failed_meta["starts"])
              and np.array_equal(failed_theta, failed_estimator.theta)
              and failed_estimator.step == 1 and not np.array_equal(failed_theta, np.full(m, .3)))
    finally:
        globals()["solve_box_qp"] = actual_solver

    source = Path(__file__)
    return dict(status="passed" if all(item["passed"] for item in checks) else "failed",
                all_passed=all(item["passed"] for item in checks),
                kind="small_algebra_and_support_enumeration_diagnostics_not_formal_experiment",
                diagnostic_seeds=[916101, 916102, 916103, 916104, 916111],
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                common_qp_sha256=hashlib.sha256(Path(fast_ewls.__file__).read_bytes()).hexdigest(),
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                setup=setup_receipt, methods=list(make_l0_kf(m)),
                max_outer_rounds_per_start=MAX_OUTER_ROUNDS,
                global_optimality_claimed=False, enumeration_cases=enumeration_cases,
                api_cases=api_cases, checks=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        parser.error("Only --check is available; this module launches no formal batch")
    report = run_diagnostics()
    destination = Path(__file__).with_name("CHECK_L0_KF.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(all_passed=report["all_passed"], checks=len(report["checks"]), output=str(destination))))
    raise SystemExit(0 if report["all_passed"] else 1)

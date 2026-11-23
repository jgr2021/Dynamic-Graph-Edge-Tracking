"""Six matched block KF point estimators using common observed statistics.

Paper: Dabush, Shlezinger, Routtenberg, arXiv:2507.09999v1, III-C/D,
https://arxiv.org/html/2507.09999v1 .  Algorithm 1 starts at the unregularized
KF mean, so its first gradient vanishes: one ISTA step is soft-thresholding.
The paper's principal experiment sets mu*rho=.25; .03/.10 are declared local
specializations, not paper parameters.  Algorithm 3 Eq.36 and official code
retain the ordinary Joseph covariance without a sparsity correction.

Official original public commit ccc8b6191670a23a7648d69122f067d83cffefca:
https://github.com/lital-dgold/PolyEKF/tree/ccc8b6191670a23a7648d69122f067d83cffefca
Its default gsp-ekf is HARD thresholding after nonnegative KF clipping, with
linear default threshold .20.  Its separate gsp-istap solves a nonnegative
L1 convex program via CVXPY/MOSEK; that is not an explicit ISTA loop.

Current-model specializations shared by every method: one Q=.02**2 I per
48-transition slow block, P0=.25**2 I, initial mean=.3, exact linear H,
common G/h supplied by structured_stats, Cholesky solves, and upper bound 1.
No transition-level Q injection, support masking, truth, or event time enters
this module.  A zero mean retains nonzero covariance and can recover later.

All means carry the UNCONSTRAINED Gaussian covariance J^-1 after their point
operation.  This is algebraically Joseph, not the exact constrained/sparse
posterior covariance; no uncertainty calibration is claimed.  The weighted
box MAP uses current-block alpha_e=.5*sqrt(2 log(max(2,m))*G_ee), a declared
strong local baseline, not the paper's scalar mu.  It calls the same compiled
solve_box_qp as fast_ewls and preserves its convergence failure metadata.

audit_state(copy=True) exposes matrices for explicit independent auditing;
normal metadata contains only scalars/strings, not m-by-m JSON duplication.
Cholesky/nonfinite arithmetic failures raise, never become successful output.
The default online path checks the actual prior/information Cholesky solves,
finite symmetric P and positive P diagonal.  An additional full P Cholesky
and J@P-I audit are available only with audit_full_covariance=True; these
unnecessary O(m**3) diagnostics are not charged to online KF baselines.
Only `python matched_kf.py --check` runs the small diagnostic suite below;
this module has no experiment/campaign mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from fast_ewls import solve_box_qp
from structured_stats import BlockStatistics


PROCESS_VARIANCE = 0.02**2
INITIAL_VARIANCE = 0.25**2
INITIAL_MEAN = 0.3
METHOD_SPECS = {
    "projected_kf": ("clip", 0.0),
    "paper_soft1_025": ("soft", 0.25),
    "soft1_003": ("soft", 0.03),
    "soft1_010": ("soft", 0.10),
    "official_hard1_020": ("hard", 0.20),
    "box_sparse_map": ("map", 0.0),
}


def _clip_soft(values, threshold):
    # Keep the paper's sign/absolute-value operation explicit before the
    # current-model [0,1] constraint.  This equals clip(values-threshold,0,1).
    soft = np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
    return np.clip(soft, 0.0, 1.0)


def _clip_hard(values, threshold):
    # Match original strict '<': equality survives the hard threshold.
    nonnegative = np.maximum(values, 0.0)
    return np.clip(np.where(nonnegative < threshold, 0.0, nonnegative), 0.0, 1.0)


def _box_kkt(J, c, theta):
    gradient = J @ theta - c
    violations = np.abs(gradient)
    lower = theta <= 1e-12
    upper = theta >= 1.0 - 1e-12
    violations[lower] = np.maximum(-gradient[lower], 0.0)
    violations[upper] = np.maximum(gradient[upper], 0.0)
    residual = float(np.max(violations))
    return residual, residual / max(1.0, float(np.max(np.abs(c))))


class MatchedKF:
    def __init__(self, m: int, name: str, *, audit_full_covariance=False):
        if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
            raise ValueError("m must be a positive integer")
        if name not in METHOD_SPECS:
            raise ValueError("Unknown matched KF method")
        self.m, self.name = int(m), name
        self.audit_full_covariance = bool(audit_full_covariance)
        self.mode, self.threshold = METHOD_SPECS[name]
        self.theta = np.full(self.m, INITIAL_MEAN)
        self.P = INITIAL_VARIANCE * np.eye(self.m)
        self.Q = PROCESS_VARIANCE * np.eye(self.m)
        self.step = 0
        self.last_prior_mean = self.last_prior_P = None
        self.last_J = self.last_rhs = self.last_alpha = None
        self.last_gaussian_mean = self.last_G = self.last_h = None

    def audit_state(self, copy=True):
        result = dict(theta=self.theta, P=self.P, Q=self.Q, step=self.step,
                      prior_mean=self.last_prior_mean, prior_P=self.last_prior_P,
                      J=self.last_J, rhs=self.last_rhs, alpha=self.last_alpha,
                      gaussian_mean=self.last_gaussian_mean,
                      G=self.last_G, h=self.last_h)
        return {key: value.copy() if copy and isinstance(value, np.ndarray) else value
                for key, value in result.items()}

    def update(self, stats: BlockStatistics):
        if not isinstance(stats, BlockStatistics):
            raise TypeError("Expected the shared structured_stats.BlockStatistics")
        G, h = np.asarray(stats.G, dtype=float), np.asarray(stats.h, dtype=float)
        if G.shape != (self.m, self.m) or h.shape != (self.m,):
            raise ValueError("Common statistics do not match the estimator dimension")
        if not np.all(np.isfinite(G)) or not np.all(np.isfinite(h)) or stats.row_count <= 0:
            raise ValueError("Finite nonempty common statistics required")
        asymmetry = float(np.max(np.abs(G - G.T)))
        if asymmetry > 1e-10 * max(1.0, float(np.max(np.abs(G)))):
            raise ValueError("Common observation Gram matrix must be symmetric")
        G = (G + G.T) * 0.5
        identity = np.eye(self.m)
        prior_mean = self.theta.copy()
        prior_P = self.P + self.Q  # exactly ONE prediction for this slow block
        prior_factor = cho_factor(prior_P, lower=True, check_finite=True)
        prior_precision = cho_solve(prior_factor, identity, check_finite=False)
        J = prior_precision + G
        J = (J + J.T) * 0.5
        rhs = prior_precision @ prior_mean + h
        factor = cho_factor(J, lower=True, check_finite=True)
        # Shared factorization for Gaussian mean and ordinary KF covariance.
        solved = cho_solve(factor, np.column_stack((rhs, identity)), check_finite=False)
        gaussian_mean = solved[:, 0]
        P = (solved[:, 1:] + solved[:, 1:].T) * 0.5
        if not np.all(np.isfinite(P)) or np.any(np.diag(P) <= 0):
            raise np.linalg.LinAlgError("Nonfinite KF covariance or nonpositive diagonal")
        linear_residual = float(np.linalg.norm(J @ gaussian_mean - rhs, np.inf))
        rhs_scale = max(1.0, float(np.linalg.norm(rhs, np.inf)))
        backward_scale = max(1.0, float(np.linalg.norm(J, np.inf)
                                       * np.linalg.norm(gaussian_mean, np.inf)
                                       + np.linalg.norm(rhs, np.inf)))
        covariance_inverse_residual = covariance_cholesky_min_diagonal = None
        if self.audit_full_covariance:
            covariance_factor = cho_factor(P, lower=True, check_finite=True)
            covariance_cholesky_min_diagonal = float(np.min(np.diag(covariance_factor[0])))
            covariance_inverse_residual = float(np.linalg.norm(J @ P - identity, np.inf))
        alpha = np.zeros(self.m)

        if self.mode == "map":
            alpha = 0.5 * np.sqrt(2.0 * np.log(max(2, self.m))
                                  * np.maximum(np.diag(G), 0.0))
            theta, meta = solve_box_qp(J, rhs - alpha, gaussian_mean)
            meta = dict(meta)
        else:
            if self.mode == "soft":
                theta = _clip_soft(gaussian_mean, self.threshold)
                optimizer = "gaussian_cholesky_then_one_soft_threshold_then_box_clip"
            elif self.mode == "hard":
                theta = _clip_hard(gaussian_mean, self.threshold)
                optimizer = "gaussian_cholesky_then_nonnegative_hard_threshold_then_box_clip"
            else:
                theta = np.clip(gaussian_mean, 0.0, 1.0)
                optimizer = "gaussian_cholesky_then_box_clip"
            residual, relative = _box_kkt(J, rhs, theta)
            meta = dict(optimizer=optimizer, iterations=1, converged=True,
                        success=True, failure_reason=None, kkt_residual=residual,
                        kkt_relative_residual=relative, kkt_tolerance=None,
                        objective=float(0.5 * theta @ J @ theta - rhs @ theta),
                        kkt_not_optimizer_gate=True,
                        kkt_target="unregularized_box_quadratic_diagnostic_only")

        theta = np.asarray(theta, dtype=float)
        finite = bool(theta.shape == (self.m,) and np.all(np.isfinite(theta))
                      and np.all(np.isfinite(P)) and np.isfinite(linear_residual)
                      and (covariance_inverse_residual is None
                           or np.isfinite(covariance_inverse_residual)))
        legal = bool(finite and np.all((theta >= 0.0) & (theta <= 1.0)))
        if not finite:
            raise FloatingPointError("Nonfinite matched KF arithmetic or solver output")
        if not legal:
            # Do not quietly clip a failed QP: that would obscure the solver.
            raise FloatingPointError("Matched KF point update violates the shared box")
        self.theta, self.P = theta.copy(), P
        self.step += 1
        self.last_prior_mean, self.last_prior_P = prior_mean, prior_P
        self.last_J, self.last_rhs, self.last_alpha = J, rhs, alpha
        self.last_gaussian_mean = gaussian_mean.copy()
        self.last_G, self.last_h = G.copy(), h.copy()
        meta.update(
            method=self.name, step=self.step, threshold=float(self.threshold),
            threshold_kind=self.mode, gaussian_solver="shared_cholesky_information_solve",
            linear_solve_residual=linear_residual,
            gaussian_relative_solve_residual=linear_residual / rhs_scale,
            gaussian_backward_error=linear_residual / backward_scale,
            covariance_inverse_residual=covariance_inverse_residual,
            covariance_spd=True if self.audit_full_covariance else None,
            covariance_spd_interpretation="checked_full_P_cholesky" if self.audit_full_covariance
                else "by_construction_information_inverse_not_full_P_chol_checked",
            covariance_full_cholesky_checked=self.audit_full_covariance,
            covariance_diagonal_min=float(np.min(np.diag(P))),
            prior_covariance_spd=True, information_spd=True,
            covariance_cholesky_min_diagonal=covariance_cholesky_min_diagonal,
            covariance_symmetry_error=float(np.max(np.abs(P - P.T))),
            process_variance_per_slow_block=PROCESS_VARIANCE,
            process_covariance_injections_this_block=1,
            measurement_transitions_in_block=int(stats.responses.shape[0]),
            block_rows=int(stats.row_count),
            block_residual_squared=stats.residual_squared(self.theta, stable=True),
            l1_alpha_min=float(np.min(alpha)), l1_alpha_max=float(np.max(alpha)),
            exact_zero_edges=int(np.count_nonzero(self.theta == 0.0)),
            changed_from_gaussian_count=int(np.count_nonzero(self.theta != gaussian_mean)),
            triggered=False, posterior_covariance_claimed=False,
            covariance_interpretation="unconstrained_gaussian_carried_after_point_update",
            covariance_accounts_for_ista_or_box=False,
            shared_observation_statistics=True, gram_recomputed_in_update=False,
            common_qp_solver=self.mode == "map",
            one_step_full_sparse_map_solved=False if self.mode == "soft" else None,
        )
        if self.mode == "map":
            meta.update(kkt_not_optimizer_gate=False,
                        kkt_target="weighted_l1_box_map",
                        alpha_source="fixed_half_current_block_score_scale_not_paper_mu")
        # A returned QP failure remains a failure; do not overwrite success or
        # converged merely because its last iterate and covariance are finite.
        return self.theta.copy(), meta


def make_kf(m, *, audit_full_covariance=False):
    """Return the six predeclared estimators, all initialized identically."""
    return {name: MatchedKF(m, name, audit_full_covariance=audit_full_covariance)
            for name in METHOD_SPECS}


def run_diagnostics():
    """Independent finite algebra/solver checks; seeds are diagnostic 914xxx.

    These random designs and off-box test targets are solver diagnostics, not
    graph-generator development/confirmation streams or performance evidence.
    """
    import scipy
    from scipy.linalg import cholesky, solve, solve_triangular
    from scipy.optimize import lsq_linear
    import fast_ewls
    from structured_stats import EdgeObservationStructure

    setup_receipt = fast_ewls.setup()
    checks = []

    def record(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def design(seed, n, transitions, target=None):
        rng = np.random.default_rng(seed)
        edges = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
        structure = EdgeObservationStructure(n, edges)
        B = np.zeros((n, len(edges)))
        B[edges[:, 0], np.arange(len(edges))] = 1.0
        B[edges[:, 1], np.arange(len(edges))] = -1.0
        states = rng.normal(size=(transitions, n))
        eta, sigma = 0.8 / n, 0.03
        H = np.vstack([eta * B * (B.T @ state)[None, :] for state in states])
        if target is None:
            target = rng.uniform(-0.25, 1.35, len(edges))
        responses = (H @ target + sigma * rng.normal(size=transitions * n)).reshape(transitions, n)
        return structure.statistics(states, responses, eta, sigma), H

    methods = make_kf(10)
    record("six_frozen_keys_and_initialization",
           list(methods) == list(METHOD_SPECS)
           and all(np.array_equal(obj.theta, np.full(10, .3))
                   and np.array_equal(obj.P, .25**2 * np.eye(10))
                   and np.array_equal(obj.Q, .02**2 * np.eye(10))
                   for obj in methods.values()), methods=list(methods))

    # Dense row-space gain/Joseph is an independent reference to the m-space
    # information update and uses the exact same observed rows.
    obj = methods["projected_kf"]
    gain_mean_errors, joseph_errors, posterior_errors = [], [], []
    independent_inverse_errors, independent_P_chol_minima = [], []
    for t in range(3):
        stats, H = design(914031 + t, 5, 9 + t)
        prior_mean = obj.theta.copy()
        prior_P = obj.P + .02**2 * np.eye(obj.m)
        R = stats.sigma**2 * np.eye(H.shape[0])
        S = H @ prior_P @ H.T + R
        K = solve(S, H @ prior_P, assume_a="pos").T
        reference_mean = prior_mean + K @ (stats.responses.ravel() - H @ prior_mean)
        left = np.eye(obj.m) - K @ H
        reference_P = left @ prior_P @ left.T + K @ R @ K.T
        theta, meta = obj.update(stats)
        audit = obj.audit_state()
        independent_P_chol_minima.append(float(np.min(np.diag(cholesky(audit["P"], lower=True)))))
        independent_inverse_errors.append(float(np.linalg.norm(audit["J"] @ audit["P"] - np.eye(obj.m), np.inf)))
        gain_mean_errors.append(float(np.max(np.abs(reference_mean - audit["gaussian_mean"]))))
        joseph_errors.append(float(np.max(np.abs(reference_P - audit["P"]))))
        posterior_errors.append(float(np.max(np.abs(theta - np.clip(reference_mean, 0, 1)))))
        json.dumps(meta, allow_nan=False)
    record("explicit_gain_and_joseph_three_blocks",
           max(gain_mean_errors) < 2e-11 and max(joseph_errors) < 2e-12
           and max(posterior_errors) < 2e-11,
           max_gaussian_mean_abs_error=max(gain_mean_errors),
           max_joseph_covariance_abs_error=max(joseph_errors),
           max_clipped_mean_abs_error=max(posterior_errors),
           independent_full_P_cholesky_min_diagonal=min(independent_P_chol_minima),
           independent_max_JP_minus_I_residual=max(independent_inverse_errors))

    default_obj = MatchedKF(10, "projected_kf")
    full_audit_obj = MatchedKF(10, "projected_kf", audit_full_covariance=True)
    audit_stats, _ = design(914034, 5, 11)
    default_theta, default_meta = default_obj.update(audit_stats)
    full_theta, full_meta = full_audit_obj.update(audit_stats)
    record("optional_full_covariance_audit_changes_diagnostics_only",
           np.array_equal(default_theta, full_theta) and np.array_equal(default_obj.P, full_audit_obj.P)
           and default_meta["covariance_spd"] is None
           and default_meta["covariance_full_cholesky_checked"] is False
           and default_meta["covariance_inverse_residual"] is None
           and full_meta["covariance_spd"] is True
           and full_meta["covariance_full_cholesky_checked"] is True
           and full_meta["covariance_inverse_residual"] < 1e-12,
           full_audit_inverse_residual=full_meta["covariance_inverse_residual"])

    # Hand-written expected vectors test soft vs hard, sign, equality, upper
    # clipping, and the paper-soft THEN box order without an optimizer oracle.
    raw = np.array([-.4, 0., .03, .1, .2, .25, .3, .8, 1.3])
    expected_soft025 = np.array([0., 0., 0., 0., 0., 0., .05, .55, 1.])
    expected_soft003 = np.array([0., 0., 0., .07, .17, .22, .27, .77, 1.])
    expected_soft010 = np.array([0., 0., 0., 0., .1, .15, .2, .7, 1.])
    expected_hard020 = np.array([0., 0., 0., 0., .2, .25, .3, .8, 1.])
    threshold_error = max(
        float(np.max(np.abs(_clip_soft(raw, .25) - expected_soft025))),
        float(np.max(np.abs(_clip_soft(raw, .03) - expected_soft003))),
        float(np.max(np.abs(_clip_soft(raw, .10) - expected_soft010))),
        float(np.max(np.abs(_clip_hard(raw, .20) - expected_hard020))))
    record("hand_reference_soft_hard_sign_equality_box_order",
           threshold_error < 2e-16, max_abs_error=threshold_error,
           hard_equality_retained=bool(_clip_hard(np.array([.2]), .2)[0] == .2))

    # Verify each actual object's recurrence, not just the pure threshold map.
    point_errors = []
    for name, (mode, threshold) in METHOD_SPECS.items():
        if mode == "map":
            continue
        point_obj = MatchedKF(10, name)
        for t in range(2):
            stats, _ = design(914041 + t, 5, 12)
            independent_prior = np.linalg.inv(point_obj.P + .02**2 * np.eye(10))
            expected_gaussian = np.linalg.solve(independent_prior + stats.G,
                                               independent_prior @ point_obj.theta + stats.h)
            if mode == "clip":
                expected = [min(1., max(0., float(x))) for x in expected_gaussian]
            elif mode == "soft":
                expected = [min(1., max(0., np.sign(x) * max(abs(x) - threshold, 0.)))
                            for x in expected_gaussian]
            else:
                expected = [0. if x < threshold else min(1., float(x)) for x in expected_gaussian]
            theta, meta = point_obj.update(stats)
            point_errors.append(float(np.max(np.abs(theta - expected))))
            record("actual_point_update_" + name + "_block" + str(t),
                   meta["success"] and point_errors[-1] < 2e-11,
                   max_abs_error=point_errors[-1])

    qp_cases = []
    for t, n in enumerate((3, 4, 5, 6)):
        stats, _ = design(914061 + t, n, 11 + t)
        map_obj = MatchedKF(len(stats.h), "box_sparse_map")
        # Two consecutive blocks ensure the sparse mean is fed back, while
        # covariance remains the declared unregularized Gaussian recursion.
        for block in range(2):
            theta, meta = map_obj.update(stats)
            audit = map_obj.audit_state()
            J, c = audit["J"], audit["rhs"] - audit["alpha"]
            # J=C.T C and C.T y=c: scipy bounded least squares independently
            # solves exactly the same convex QP up to an additive constant.
            C = cholesky(J, lower=True).T
            y = solve_triangular(C.T, c, lower=True)
            independent = lsq_linear(C, y, bounds=(0., 1.), method="trf",
                                     tol=1e-13, max_iter=1000)
            error = float(np.max(np.abs(theta - independent.x)))
            objective = lambda x: float(.5 * x @ J @ x - c @ x)
            relative_objective_error = abs(objective(theta) - objective(independent.x)) / max(1., abs(objective(independent.x)))
            qp_cases.append(dict(seed=914061 + t, n=n, block=block,
                                 independent_success=bool(independent.success),
                                 common_solver_success=bool(meta["success"]),
                                 max_theta_abs_error=error,
                                 relative_objective_error=relative_objective_error,
                                 kkt_relative_residual=meta["kkt_relative_residual"]))
    record("independent_scipy_bounded_least_squares_qp",
           all(case["independent_success"] and case["common_solver_success"]
               and case["max_theta_abs_error"] < 2e-7
               and case["relative_objective_error"] < 2e-10
               for case in qp_cases), cases=qp_cases)

    # 48 consensus transitions have exactly no topology information.  Q must
    # still be injected once per CALL/block, not 48 times or per node row.
    rng = np.random.default_rng(914081)
    structure = stats.structure
    consensus = np.repeat(rng.normal(size=(48, 1)), structure.n, axis=1)
    responses = np.repeat(rng.normal(size=(48, 1)), structure.n, axis=1)
    empty_stats = structure.statistics(consensus, responses, .8 / structure.n, .03)
    no_information_cases = []
    for name, zero_obj in make_kf(structure.m).items():
        expected_mean = np.full(structure.m, .3)
        for block in range(3):
            mode, threshold = METHOD_SPECS[name]
            if mode == "soft":
                expected_mean = np.maximum(expected_mean - threshold, 0.)
            elif mode == "hard":
                expected_mean = np.where(expected_mean < threshold, 0., expected_mean)
            theta, meta = zero_obj.update(empty_stats)
            expected_P = (.25**2 + (block + 1) * .02**2) * np.eye(structure.m)
            no_information_cases.append(dict(method=name, block=block,
                theta_error=float(np.max(np.abs(theta - expected_mean))),
                covariance_error=float(np.max(np.abs(zero_obj.P - expected_P))),
                success=bool(meta["success"]),
                injections=meta["process_covariance_injections_this_block"]))
    record("no_information_48_transitions_one_Q_per_slow_block",
           np.count_nonzero(empty_stats.G) == 0 and np.count_nonzero(empty_stats.h) == 0
           and all(case["theta_error"] < 1e-12 and case["covariance_error"] < 1e-13
                   and case["success"] and case["injections"] == 1
                   for case in no_information_cases), cases=no_information_cases)

    # A zeroed mean is not a known-absent support.  All candidates retain Q/P
    # and can activate on a later independently generated informative block.
    zero_obj = MatchedKF(15, "paper_soft1_025")
    zero_obj.update(empty_stats)
    zero_obj.update(empty_stats)
    covariance_before = zero_obj.P.copy()
    revival_stats, _ = design(914091, 6, 32, np.full(15, .8))
    revived, revival_meta = zero_obj.update(revival_stats)
    record("zero_mean_edges_can_restore_without_support_oracle",
           np.all(np.diag(covariance_before) > 0)
           and np.all(revived > 0) and revival_meta["success"],
           restored_edges=int(np.count_nonzero(revived > 0)),
           min_carried_variance=float(np.min(np.diag(covariance_before))))

    # Returned audit arrays and means must not alias internal recursive state.
    audit = zero_obj.audit_state()
    protected_P, protected_J = zero_obj.P.copy(), zero_obj.last_J.copy()
    audit["P"].fill(-1.)
    audit["J"].fill(-1.)
    audit["rhs"].fill(-1.)
    audit["alpha"].fill(-1.)
    audit["gaussian_mean"].fill(-1.)
    revived.fill(-1.)
    record("audit_and_returned_arrays_are_copies",
           np.array_equal(zero_obj.P, protected_P)
           and np.array_equal(zero_obj.last_J, protected_J)
           and np.all(zero_obj.theta >= 0)
           and np.all(zero_obj.last_gaussian_mean > 0))

    # A numerically finite last QP iterate does not erase its convergence
    # failure.  This explicit diagnostic replaces only the in-memory callable.
    original_solver = globals()["solve_box_qp"]
    def failed_solver(J, c, initial):
        theta = np.clip(initial, 0., 1.)
        return theta, dict(optimizer="diagnostic_injected_failed_solver", iterations=5000,
                           converged=False, success=False, failure_reason="diagnostic_budget_exhausted",
                           kkt_residual=1., kkt_relative_residual=1., kkt_tolerance=1e-8,
                           objective=float(.5 * theta @ J @ theta - c @ theta))
    try:
        globals()["solve_box_qp"] = failed_solver
        failed_obj = MatchedKF(15, "box_sparse_map")
        _, failed_meta = failed_obj.update(revival_stats)
        record("qp_failure_is_not_overwritten_by_finite_covariance",
               failed_meta["success"] is False and failed_meta["converged"] is False
               and failed_meta["iterations"] == 5000
               and failed_meta["failure_reason"] == "diagnostic_budget_exhausted")
    finally:
        globals()["solve_box_qp"] = original_solver

    # Non-SPD covariance is an explicit numerical exception, with no claimed
    # successful update or hidden diagonal-jitter/previous-mean fallback.
    broken_obj = MatchedKF(15, "projected_kf")
    broken_obj.P = -np.eye(15)
    raised = False
    try:
        broken_obj.update(revival_stats)
    except np.linalg.LinAlgError:
        raised = True
    record("non_spd_covariance_failure_is_explicit", raised and broken_obj.step == 0)

    record("metadata_JSON_and_covariance_claim_boundary",
           all(revival_meta[key] is False for key in (
               "posterior_covariance_claimed", "covariance_accounts_for_ista_or_box"))
           and revival_meta["covariance_spd"] is None
           and revival_meta["covariance_full_cholesky_checked"] is False
           and revival_meta["information_spd"]
           and revival_meta["covariance_diagonal_min"] > 0
           and not any(isinstance(value, np.ndarray) for value in revival_meta.values()))
    json.dumps(revival_meta, allow_nan=False)
    source = Path(__file__)
    return dict(status="passed" if all(item["passed"] for item in checks) else "failed",
                all_passed=all(item["passed"] for item in checks),
                kind="finite_diagnostic_algebra_checks_no_formal_batch",
                diagnostic_seed_namespace="914xxx; independent of graph development/confirmation",
                diagnostic_seed_ids=[914031, 914032, 914033, 914034, 914041, 914042,
                                     914061, 914062, 914063, 914064, 914081, 914091],
                methods=list(METHOD_SPECS), process_variance_per_slow_block=PROCESS_VARIANCE,
                prior_variance=INITIAL_VARIANCE, prior_mean=INITIAL_MEAN,
                setup=setup_receipt, numpy_version=np.__version__, scipy_version=scipy.__version__,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                fast_ewls_sha256=hashlib.sha256(Path(fast_ewls.__file__).read_bytes()).hexdigest(),
                checks=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run only the finite diagnostic checks")
    args = parser.parse_args()
    if not args.check:
        parser.error("Only --check is provided; this module does not launch experiments")
    receipt = run_diagnostics()
    output = Path(__file__).with_name("CHECK_MATCHED_KF.json")
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(all_passed=receipt["all_passed"], checks=len(receipt["checks"]),
                          output=str(output)), ensure_ascii=False))
    raise SystemExit(0 if receipt["all_passed"] else 1)

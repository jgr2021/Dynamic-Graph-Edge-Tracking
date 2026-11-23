"""Same-Q information KF-L0 with its own unconstrained Gaussian center.

Only the natural-vector anchor changes from the original sparse point to
the own previous unconstrained Gaussian mean. Import performs no setup,
actor construction, numerical update, timing experiment or observation draw.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


ROOT = Path(__file__).resolve().parent
V16 = ROOT.parent/'continuation_20261024_fixed_star_development_v16'
if str(V16) not in sys.path:
    sys.path.insert(0, str(V16))
_l0 = importlib.import_module('l0_kf')
_statistics = importlib.import_module('structured_stats')
for module, filename in ((_l0, 'l0_kf.py'), (_statistics, 'structured_stats.py')):
    if Path(module.__file__).resolve() != (V16/filename).resolve():
        raise ImportError('V24 requires the frozen original V16 module: '+filename)

BlockStatistics = _statistics.BlockStatistics
METHOD_NAME = 'kf_l0_gaussian_center_incidence'
INITIAL_PRECISION = 16.0
INITIAL_MEAN = .3
PROCESS_VARIANCE = .0004
ANCHOR_KIND = 'own_previous_unconstrained_Gaussian_mu'


class GaussianCenterKF:
    """Own full J and unbounded finite mu; sparse theta is a readout only."""

    def __init__(self, m):
        if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
            raise ValueError('m must be a positive integer')
        self.m = int(m)
        self.name = METHOD_NAME
        self.theta = np.full(self.m, INITIAL_MEAN)
        self.mu = np.full(self.m, INITIAL_MEAN)
        self.J = np.eye(self.m)*INITIAL_PRECISION
        self.q = PROCESS_VARIANCE
        self.kappa = .5*float(np.log(max(2, self.m)))
        self.step = 0
        self.last = dict(J_previous=None, Aminus=None, G=None, h=None, rhs=None,
            gaussian_mean=None, previous_theta=None, previous_mu=None,
            actual_anchor=None, anchor_kind=ANCHOR_KIND)

    def audit_state(self, copy=True):
        state = dict(theta=self.theta, mu=self.mu, J=self.J, step=self.step,
            q=self.q, kappa=self.kappa, support=self.theta > 0, **self.last)
        return {key: value.copy() if copy and isinstance(value, np.ndarray) else value
                for key, value in state.items()}

    def update(self, stats):
        started = time.perf_counter()
        if not isinstance(stats, BlockStatistics):
            raise TypeError('Original V16 structured_stats.BlockStatistics required')
        G, h = np.asarray(stats.G, dtype=float), np.asarray(stats.h, dtype=float)
        if G.shape != (self.m, self.m) or h.shape != (self.m,) or stats.row_count <= 0:
            raise ValueError('Common statistics dimension/row count mismatch')
        if not np.all(np.isfinite(G)) or not np.all(np.isfinite(h)):
            raise ValueError('Finite common statistics required')
        if np.max(np.abs(G-G.T)) > 1e-10*max(1., float(np.max(np.abs(G)))):
            raise ValueError('Symmetric observation information required')
        if (self.theta.shape != (self.m,) or not np.all(np.isfinite(self.theta))
                or np.any((self.theta < 0) | (self.theta > 1))):
            raise FloatingPointError('Own actual previous sparse point is invalid; no reset')
        if self.mu.shape != (self.m,) or not np.all(np.isfinite(self.mu)):
            raise FloatingPointError('Own previous unconstrained Gaussian mean is invalid; no reset or clip')
        old_J = self.J
        if old_J.shape != (self.m, self.m) or not np.all(np.isfinite(old_J)):
            raise FloatingPointError('Own carried full precision must be finite')
        if np.max(np.abs(old_J-old_J.T)) > 1e-10*max(1., float(np.max(np.abs(old_J)))):
            raise FloatingPointError('Own carried full precision must be symmetric')
        G = (G+G.T)*.5
        previous_theta = self.theta.copy()
        anchor = self.mu.copy()
        normalization_s = time.perf_counter()-started

        prediction_started = time.perf_counter()
        system = np.eye(self.m)+self.q*old_J
        Aminus = cho_solve(cho_factor(system, lower=True, check_finite=True),
                           old_J, check_finite=False)
        Aminus = (Aminus+Aminus.T)*.5
        if not np.all(np.isfinite(Aminus)):
            raise FloatingPointError('Nonfinite own Riccati precision; no ridge or reset')
        J = Aminus+G
        J = (J+J.T)*.5
        # The only model-recursion change from the original V18 actor.
        b = Aminus@anchor+h
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(b)):
            raise FloatingPointError('Nonfinite own working precision or natural vector')
        prediction_s = time.perf_counter()-prediction_started

        gaussian_started = time.perf_counter()
        factor = cho_factor(J, lower=True, check_finite=True)
        gaussian = cho_solve(factor, b, check_finite=False)
        mean_residual = float(np.linalg.norm(J@gaussian-b, np.inf))
        backward_scale = max(1., float(np.linalg.norm(J, np.inf)*np.linalg.norm(gaussian, np.inf)
                                      + np.linalg.norm(b, np.inf)))
        backward_error = mean_residual/backward_scale
        if (not np.all(np.isfinite(gaussian)) or not np.isfinite(mean_residual)
                or not np.isfinite(backward_scale) or not np.isfinite(backward_error)):
            raise FloatingPointError('Nonfinite own Gaussian solve diagnostics')
        gaussian_s = time.perf_counter()-gaussian_started

        point_started = time.perf_counter()
        theta, meta = _l0.solve_l0_box(J, b, self.kappa, gaussian)
        point_s = time.perf_counter()-point_started

        carry_started = time.perf_counter()
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.m,) or not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > 1)):
            raise FloatingPointError('Frozen L0 solver returned invalid actual point')
        # A failed local branch remains failed; no sparse feedback into mu.
        self.theta, self.mu, self.J = theta.copy(), gaussian.copy(), J
        self.step += 1
        self.last = dict(J_previous=old_J, Aminus=Aminus, G=G.copy(), h=h.copy(),
            rhs=b.copy(), gaussian_mean=gaussian.copy(), previous_theta=previous_theta,
            previous_mu=anchor.copy(), actual_anchor=anchor, anchor_kind=ANCHOR_KIND)
        carry_s = time.perf_counter()-carry_started
        meta.update(method=self.name, step=self.step, kappa=self.kappa,
            kappa_policy='.5*log(max(2,m))',
            fixed_start_order=['clipped_gaussian', 'all_zero'],
            max_outer_rounds_per_start=_l0.MAX_OUTER_ROUNDS,
            no_physical_gap_prior=True, exact_zero_edges=int(np.count_nonzero(self.theta == 0)),
            linear_solve_residual=mean_residual, gaussian_backward_error=backward_error,
            gaussian_backward_scale=backward_scale,
            gaussian_rhs_scale=max(1., float(np.linalg.norm(b, np.inf))),
            gaussian_rhs_relative_residual=mean_residual/max(1., float(np.linalg.norm(b, np.inf))),
            gaussian_numeric_threshold_online=None,
            gaussian_residual_policy='same_V18_finite_and_report; numeric_tolerance_offline',
            gaussian_solver='dense_cholesky_single_rhs', gaussian_rhs_count=1,
            prediction='exact_Riccati_on_own_full_precision',
            prediction_system_cholesky_required=True,
            prediction_full_matrix_backward_check_online=False,
            prediction_full_matrix_backward_check_scope='separate_offline_qualification',
            information_spd=True, information_spd_interpretation='working_J_Cholesky_passed',
            predicted_precision_spd_interpretation='Riccati_of_own_positive_precision_by_construction; full_check_offline',
            full_covariance_carried=False, current_full_covariance_formed=False,
            stored_dense_edge_precision=True, matrix_free_information=False,
            full_update_subcubic_claimed=False, subquadratic_storage_claimed=False,
            posterior_covariance_claimed=False, covariance_accounts_for_support_selection=False,
            global_optimality_claimed=False, novelty_claimed=False,
            support_rows_permanently_masked=False, all_candidates_remain_available=True,
            process_variance_per_slow_block=self.q, process_covariance_injections_this_block=1,
            initial_mean=INITIAL_MEAN, initial_precision=INITIAL_PRECISION,
            block_rows=int(stats.row_count), block_residual_squared=stats.residual_squared(self.theta, stable=True),
            triggered=False, shared_observation_statistics=True,
            gram_recomputed_in_update=False, common_qp_solver=True, no_truth_input=True,
            own_history_only=True, uses_truth_support=False, uses_event_times=False,
            anchor_kind=ANCHOR_KIND, gaussian_center_used_in_prediction=True,
            mu_carried_unconstrained=True, mu_is_working_gaussian_not_sparse_posterior=True,
            sparse_point_reanchors_history=False, mu_clip_or_reset_applied=False,
            input_normalization_seconds=float(normalization_s),
            subcosts=dict(prediction_seconds=float(prediction_s), gaussian_seconds=float(gaussian_s),
                point_solve_seconds=float(point_s), carry_seconds=float(carry_s)))
        meta['update_seconds_internal'] = float(time.perf_counter()-started)
        return self.theta.copy(), meta

"""Independent first-order author-formula adaptations; admitted import only.

No author repository module is imported or copied. The caller fixes tau and
all experiment choices. Normal formal updates do not compute predictive NLL.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.linalg import cholesky, cho_solve, eigh, svd

INITIAL_MEAN = .3
INITIAL_PRECISION = 16.
PROCESS_VARIANCE = .0004
SVD_RELATIVE_CUTOFF = .001
ROUNDOFF_MULTIPLIER = 128.


def _real_array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name+' must be real')
    result = np.array(value, dtype=np.float64, order='C', copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError('Finite '+name+' with actual dimensions required')
    return result


def _symmetric(value, name):
    error = float(np.max(np.abs(value-value.T)))
    if error > 1e-10*max(1., float(np.max(np.abs(value)))):
        raise ValueError(name+' must be symmetric')
    return np.ascontiguousarray(.5*(value+value.T)), error


class Actor:
    """Working Gaussian P with its own processed nonnegative point feedback.

    Public theta/P may be replaced before step0 in artificial checks. Formal
    initialization is fixed by protocol and independently audited.
    """
    def __init__(self, m, kind='hard', tau=None, box=False):
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError('Positive full candidate-edge dimension required')
        if kind not in ('hard', 'soft'):
            raise ValueError('kind must be hard or soft')
        if tau is None or isinstance(tau, bool) or not np.isscalar(tau) or not np.isfinite(tau) or tau < 0.:
            raise ValueError('Explicit finite nonnegative tau required')
        if not isinstance(box, (bool, np.bool_)):
            raise ValueError('Explicit Boolean box policy required')
        self.m, self.kind, self.tau, self.box = int(m), kind, float(tau), bool(box)
        self.name = kind+('_box' if self.box else '_native')
        self.theta = np.full(self.m, INITIAL_MEAN)
        self.P = np.eye(self.m)/INITIAL_PRECISION
        self.q = PROCESS_VARIANCE
        self.step = 0
        self.last = {}
        self.initialization_matches_protocol = None

    def audit_state(self, copy=True):
        result = dict(theta=self.theta, P=self.P, step=self.step, kind=self.kind,
                      tau=self.tau, box=self.box, process_variance=self.q, **self.last)
        return {key: value.copy() if copy and isinstance(value, np.ndarray) else value
                for key, value in result.items()}

    def update(self, stats, dense_loader=None, *, compute_score=False):
        required = ('G', 'h', 'response_norm_squared_scaled', 'row_count', 'sigma')
        if any(not hasattr(stats, key) for key in required):
            raise TypeError('Observed sufficient statistics, dimension and sigma required')
        residual = getattr(stats, 'residual', None)
        return self.update_from_arrays(stats.G, stats.h, stats.response_norm_squared_scaled,
            stats.row_count, stats.sigma, dense_loader=dense_loader,
            compute_score=compute_score, residual_func=residual if callable(residual) else None)

    def update_from_arrays(self, G, h, response_norm_squared_scaled, dobs, sigma,
                           dense_loader=None, *, compute_score=False, residual_func=None):
        started = time.perf_counter()
        m = self.m
        G = _real_array(G, (m, m), 'G'); G, symmetry = _symmetric(G, 'G')
        h = _real_array(h, (m,), 'h')
        if (isinstance(dobs, bool) or not isinstance(dobs, (int, np.integer)) or dobs < 1
                or not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0.
                or not np.isscalar(response_norm_squared_scaled)
                or not np.isfinite(response_norm_squared_scaled) or response_norm_squared_scaled < 0.):
            raise ValueError('Positive observation count/sigma and finite nonnegative observed norm required')
        if not isinstance(compute_score, (bool, np.bool_)):
            raise ValueError('compute_score must be Boolean')
        dobs, sigma, c0 = int(dobs), float(sigma), float(response_norm_squared_scaled)
        previous_theta = _real_array(self.theta, (m,), 'own previous theta')
        previous_P = _real_array(self.P, (m, m), 'own previous working P')
        previous_P, previous_symmetry = _symmetric(previous_P, 'own previous working P')
        if np.any(previous_theta < 0.) or self.box and np.any(previous_theta > 1.):
            raise FloatingPointError('Own processed previous theta violates declared point domain')
        if self.step == 0:
            self.initialization_matches_protocol = bool(
                np.array_equal(previous_theta, np.full(m, INITIAL_MEAN))
                and np.array_equal(previous_P, np.eye(m)/INITIAL_PRECISION))
        input_seconds = time.perf_counter()-started
        clock = time.perf_counter()
        mu_prior = previous_theta.copy()
        P_prior = previous_P+self.q*np.eye(m)
        L = cholesky(P_prior, lower=True, check_finite=True)
        prediction_seconds = time.perf_counter()-clock
        dense_cache = [None]
        dense_load_seconds = [0.]
        dense_errors = {}

        def load_dense():
            if dense_cache[0] is None:
                if dense_loader is None:
                    raise FloatingPointError('Mask boundary requires paid raw dense_loader(H,y,sigma)')
                load_clock = time.perf_counter()
                raw = dense_loader()
                if not isinstance(raw, (tuple, list)) or len(raw) != 3:
                    raise ValueError('dense_loader must return raw H,y,sigma')
                H = _real_array(raw[0], (dobs, m), 'raw dense H')
                yraw = np.asarray(raw[1])
                if np.iscomplexobj(yraw) or yraw.size != dobs:
                    raise ValueError('Raw dense response dimensions differ')
                y = _real_array(yraw.reshape(dobs), (dobs,), 'raw dense y')
                if not np.isscalar(raw[2]) or float(raw[2]) != sigma:
                    raise ValueError('Dense and sufficient-statistic sigma must match')
                Hw, yw = H/sigma, y/sigma
                actual_G, actual_h, actual_c0 = Hw.T@Hw, Hw.T@yw, float(yw@yw)
                for name, actual, expected in (('G', actual_G, G), ('h', actual_h, h),
                                               ('response_norm_squared_scaled', actual_c0, c0)):
                    scale = max(1., float(np.max(np.abs(np.asarray(expected)))))
                    tolerance = 256.*np.finfo(float).eps*max(dobs, m)*scale
                    error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
                    if error > tolerance:
                        raise FloatingPointError('Dense loader differs from actual '+name)
                    dense_errors[name+'_max_abs_error'] = error
                    dense_errors[name+'_tolerance'] = tolerance
                dense_cache[0] = H, y
                dense_load_seconds[0] += time.perf_counter()-load_clock
            return dense_cache[0]

        clock = time.perf_counter()
        C = L.T@G@L; C, C_symmetry = _symmetric(C, 'whitened state Gram')
        natural_residual = h-G@mu_prior
        v = L.T@natural_residual
        eigen_raw = eigenvalues = eigenvectors = spectrum = state_mask = margins = covariance_factors = None
        dense_singular_values = dense_mask = None
        d_tolerance = d_min_raw = d_min = minimum_margin = noise_margin = compressed_cutoff = None
        cutoff_raw = cutoff_white = None
        negative_clamped_count = 0
        fallback = False; fallback_reason = None; dense_svd_seconds = 0.
        exact_factor = exact_u = exact_z = None
        if self.kind == 'hard':
            eigen_raw, eigenvectors = eigh(C, check_finite=True)
            d_tolerance = float(ROUNDOFF_MULTIPLIER*np.finfo(float).eps*m
                                *max(1., float(np.max(np.abs(eigen_raw)))))
            d_min_raw = float(eigen_raw[0])
            if d_min_raw < -d_tolerance:
                raise FloatingPointError('Whitened Gram has negative eigenvalue beyond fixed roundoff budget')
            eigenvalues = eigen_raw.copy()
            negative_clamped_count = int(np.count_nonzero(eigenvalues < 0.))
            eigenvalues[eigenvalues < 0.] = 0.
            d_min = float(eigenvalues[0]); spectrum = 1.+eigenvalues
            compressed_cutoff = float(SVD_RELATIVE_CUTOFF*max(1., float(spectrum[-1])))
            state_mask = spectrum > compressed_cutoff
            margins = np.abs(spectrum-compressed_cutoff)
            minimum_margin = float(np.min(margins))
            if dobs > m:
                noise_margin = abs(1.-compressed_cutoff)
                minimum_margin = min(minimum_margin, noise_margin)
            fallback = bool(minimum_margin <= d_tolerance)
            cutoff_white = compressed_cutoff; cutoff_raw = sigma*sigma*cutoff_white
            if fallback:
                fallback_reason = 'innovation_mask_roundoff_margin'
                H, y = load_dense()
                dense_clock = time.perf_counter()
                observed = H@P_prior@H.T
                S = .5*(observed+observed.T)+(sigma*sigma)*np.eye(dobs)
                U, dense_singular_values, Vt = svd(S, full_matrices=False, check_finite=True)
                cutoff_raw = float(SVD_RELATIVE_CUTOFF*dense_singular_values[0])
                cutoff_white = cutoff_raw/(sigma*sigma)
                dense_mask = dense_singular_values > cutoff_raw
                reciprocal = np.zeros_like(dense_singular_values)
                reciprocal[dense_mask] = 1./dense_singular_values[dense_mask]
                inverse = (Vt.T*reciprocal)@U.T
                gain = P_prior@H.T@inverse
                z = mu_prior+gain@(y-H@mu_prior)
                joseph = np.eye(m)-gain@H
                P = joseph@P_prior@joseph.T+(sigma*sigma)*(gain@gain.T)
                dense_svd_seconds = time.perf_counter()-dense_clock
                covariance_factors = np.where(state_mask, 1./spectrum, 1.)
                gaussian_mode = 'paid_dense_observation_svd_joseph'
            else:
                coefficients = state_mask.astype(float)/spectrum
                z = mu_prior+L@(eigenvectors@(coefficients*(eigenvectors.T@v)))
                covariance_factors = np.where(state_mask, 1./spectrum, 1.)
                root = L@eigenvectors
                P = (root*covariance_factors)@root.T
                gaussian_mode = 'state_compressed_strict_svd_joseph_equivalent'
        else:
            A = np.eye(m)+C
            exact_factor = cholesky(A, lower=True, check_finite=True)
            exact_u = cho_solve((exact_factor, True), v, check_finite=False)
            exact_z = mu_prior+L@exact_u
            z = exact_z.copy()
            P = L@cho_solve((exact_factor, True), L.T, check_finite=False)
            gaussian_mode = 'untruncated_exact_gaussian_cholesky'
        P, posterior_symmetry = _symmetric(P, 'posterior working P')
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(P)) or np.any(np.diag(P) <= 0.):
            raise FloatingPointError('Invalid working Gaussian update')
        gaussian_seconds = time.perf_counter()-clock

        score_fields = dict(predictive_nll=None, predictive_score=None, predictive_score_per_observation=None,
            score_logdet=None, score_gaussian_quadratic=None,
            score_gaussian_quadratic_analytic=None, score_gaussian_quadratic_centered=None,
            score_analytic_minus_stable=None, score_innovation_norm_squared_scaled=None,
            score_innovation_norm_squared_scaled_analytic=None, score_centering_difference=None,
            score_common_constant=None, score_source=None)
        score_seconds = 0.; score_z = None
        if compute_score:
            clock = time.perf_counter()
            if exact_factor is None:
                exact_factor = cholesky(np.eye(m)+C, lower=True, check_finite=True)
                exact_u = cho_solve((exact_factor, True), v, check_finite=False)
                exact_z = mu_prior+L@exact_u
            score_z = exact_z.copy()
            analytic_c = float(math.fsum([c0, -2.*float(mu_prior@h), float(mu_prior@G@mu_prior)]))
            correction = float(v@exact_u)
            if residual_func is not None:
                r0 = np.asarray(residual_func(mu_prior), dtype=float).reshape(-1)
                r1 = np.asarray(residual_func(exact_z), dtype=float).reshape(-1)
                if r0.shape != (dobs,) or r1.shape != (dobs,) or not np.all(np.isfinite(r0)) or not np.all(np.isfinite(r1)):
                    raise FloatingPointError('Observed stable residual is not finite with actual dimension')
                c = float((r0/sigma)@(r0/sigma))
                quadratic = float((r1/sigma)@(r1/sigma)+exact_u@exact_u)
                score_source = 'paid_stable_node_augmented_residual'
            elif dense_loader is not None:
                H, y = load_dense()
                r0, r1 = (y-H@mu_prior)/sigma, (y-H@exact_z)/sigma
                c = float(r0@r0); quadratic = float(r1@r1+exact_u@exact_u)
                score_source = 'paid_stable_dense_augmented_residual'
            else:
                c = analytic_c; quadratic = c-correction
                score_source = 'sufficient_statistics_analytic_no_clipping'
                if c < 0. or quadratic < 0.:
                    raise FloatingPointError('Analytic predictive quadratic cancellation; stable observed residual required')
            logdet = float(2.*np.log(np.diag(exact_factor)).sum())
            common_constant = float(dobs*math.log(2.*math.pi*sigma*sigma))
            score = .5*(common_constant+logdet+quadratic)
            if not all(np.isfinite(x) for x in [analytic_c, c, quadratic, logdet, common_constant, score]) or quadratic < 0.:
                raise FloatingPointError('Invalid predictive Gaussian integral; NLL itself may be negative')
            score_fields.update(predictive_nll=float(score), predictive_score=float(score), predictive_score_per_observation=float(score/dobs),
                score_logdet=logdet, score_gaussian_quadratic=quadratic,
                score_gaussian_quadratic_analytic=float(analytic_c-correction),
                score_gaussian_quadratic_centered=float(c-correction),
                score_analytic_minus_stable=float(analytic_c-correction-quadratic),
                score_innovation_norm_squared_scaled=c,
                score_innovation_norm_squared_scaled_analytic=analytic_c,
                score_centering_difference=float(analytic_c-c), score_common_constant=common_constant,
                score_source=score_source)
            score_seconds = time.perf_counter()-clock

        clock = time.perf_counter()
        if self.kind == 'hard':
            theta = np.maximum(z, 0.)
            theta[theta < self.tau] = 0.
            point_rule = 'gaussian_then_NN_then_strict_less_than_tau_zero_equal_retained'
        else:
            theta = np.maximum(z-self.tau, 0.)
            point_rule = 'untruncated_gaussian_then_positive_soft_threshold'
        if self.box:
            theta = np.minimum(theta, 1.)
        if not np.all(np.isfinite(theta)) or np.any(theta < 0.) or self.box and np.any(theta > 1.):
            raise FloatingPointError('Invalid processed point for declared domain')
        point_seconds = time.perf_counter()-clock
        clock = time.perf_counter()
        self.theta, self.P = theta.copy(), P.copy()
        self.last = dict(previous_theta=previous_theta, P_previous=previous_P,
            mu_prior=mu_prior, P_prior=P_prior, z=z.copy(), G=G, h=h,
            whitened_Gram=C, prior_cholesky=L,
            state_eigenvalues_raw=eigen_raw, state_eigenvalues=eigenvalues,
            state_eigenvectors=eigenvectors, innovation_eigenvalues_white=spectrum,
            innovation_mask_state=state_mask, mask_margins_white=margins,
            covariance_factors_state=covariance_factors, score_exact_gaussian_z=score_z)
        self.step += 1
        carry_seconds = time.perf_counter()-clock
        meta = dict(method=self.name, kind=self.kind, tau=self.tau, box=self.box,
            step=self.step, success=True, actual_prior_center='own_previous_processed_theta',
            initial_mean=INITIAL_MEAN, initial_precision=INITIAL_PRECISION,
            initialization_matches_protocol=self.initialization_matches_protocol,
            process_variance_per_slow_block=self.q, process_covariance_injections_this_block=1,
            t0_process_covariance_included=True, point_feedback_into_prediction=True,
            working_covariance_not_selected_posterior=True, native_upper_bound=None if not self.box else 1.,
            point_rule=point_rule, gaussian_mode=gaussian_mode,
            svd_relative_cutoff=SVD_RELATIVE_CUTOFF if self.kind == 'hard' else None,
            innovation_mask_comparison='strict_greater_than' if self.kind == 'hard' else None,
            cutoff_raw=cutoff_raw, cutoff_white=cutoff_white, compressed_cutoff_white=compressed_cutoff,
            d_min_raw=d_min_raw, d_min_after_clamp=d_min, d_clamp_tolerance=d_tolerance,
            d_negative_clamped_count=negative_clamped_count, mask_min_margin_white=minimum_margin,
            extra_noise_only_multiplicity=max(0, dobs-m) if self.kind == 'hard' else None,
            noise_only_margin_white=noise_margin, dense_fallback=fallback,
            dense_fallback_reason=fallback_reason, dense_loader_called=dense_cache[0] is not None,
            dense_innovation_singular_values=None if dense_singular_values is None else dense_singular_values.tolist(),
            dense_innovation_mask=None if dense_mask is None else dense_mask.tolist(),
            dense_consistency=dense_errors, Gram_symmetry_error=symmetry,
            prior_covariance_symmetry_error=previous_symmetry, whitened_Gram_symmetry_error=C_symmetry,
            posterior_covariance_symmetry_error=posterior_symmetry,
            response_norm_squared_scaled=c0, block_rows=dobs, sigma=sigma,
            score_computed=bool(compute_score), score_is_preupdate_prior_integral=bool(compute_score),
            score_uses_untruncated_exact_mode=bool(compute_score), score_clipping_applied=False,
            formula_implementation='independent_first_order_adaptation_no_author_code_import',
            full_posterior_or_calibration_claimed=False,
            input_seconds=float(input_seconds), prediction_seconds=float(prediction_seconds),
            gaussian_update_seconds=float(gaussian_seconds), score_seconds=float(score_seconds),
            point_seconds=float(point_seconds), carry_seconds=float(carry_seconds),
            dense_load_seconds=float(dense_load_seconds[0]), dense_svd_seconds=float(dense_svd_seconds),
            dense_timings_are_subsets_not_additive=True, **score_fields)
        meta['update_seconds_internal'] = float(time.perf_counter()-started)
        return self.theta.copy(), meta

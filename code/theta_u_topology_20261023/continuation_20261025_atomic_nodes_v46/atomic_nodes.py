"""Finite atomic-edge mean-field filter; definitions only, no setup on import.

All states are legal nonnegative edge weights.  Discrete quadrature, independent
edge prediction, mean-field measurement updates and a support decision are
declared approximations, not calibrated continuous graph Bayes.  Runtime imports
this module only after the owning parent has admitted its single finite batch.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.special import log_ndtr, logsumexp
from numba import njit

MAX_SWEEPS = 30
TOLERANCE = 1e-8
BIRTH_PROBABILITY = .01
DEATH_PROBABILITY = .01
PROCESS_SD = .02
INITIAL_ATOM_PROBABILITY = .5
POSITIVE_PRIOR_MEAN = .3
POSITIVE_PRIOR_SD = .25
PRIMARY_ACTION = 'factorized_support_MAP_conditional_mean'
SECONDARY_ACTION = 'posterior_mean'
_DEFAULT_MODEL_CACHE = None


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise FloatingPointError('Nonfinite atomic-node ' + name)


def _log_difference(first, second):
    """log(exp(first)-exp(second)); no probability floor."""
    if second > first:
        if second-first > 1e-13:
            raise FloatingPointError('CDF interval order is reversed')
        raise FloatingPointError('CDF interval lost its declared positive width')
    if second == -np.inf:
        return float(first)
    if first == second:
        return -np.inf
    return float(first + np.log(-np.expm1(second-first)))


def _normal_interval_log_mass(lower, upper):
    if not lower < upper:
        raise ValueError('Positive interval width is required')
    if lower >= 0.:
        return _log_difference(float(log_ndtr(-lower)), float(log_ndtr(-upper)))
    return _log_difference(float(log_ndtr(upper)), float(log_ndtr(lower)))


def _positive_grid_logs(edges, location, scale):
    if not scale > 0.:
        raise ValueError('Positive Gaussian scale required')
    logs = np.asarray([_normal_interval_log_mass((a-location)/scale,
        (b-location)/scale) for a, b in zip(edges[:-1], edges[1:])])
    if not np.all(np.isfinite(logs)):
        raise FloatingPointError('Positive-grid interval mass was not resolved')
    logs -= float(logsumexp(logs))
    return logs


def default_model():
    global _DEFAULT_MODEL_CACHE
    if _DEFAULT_MODEL_CACHE is not None:
        return {name: value.copy() for name, value in _DEFAULT_MODEL_CACHE.items()}
    values = np.arange(65, dtype=float)/64.
    positive = values[1:]
    edges = np.concatenate(([0.], .5*(positive[:-1]+positive[1:]), [1.]))
    birth_logs = _positive_grid_logs(edges, POSITIVE_PRIOR_MEAN, POSITIVE_PRIOR_SD)
    log_transition = np.empty((65, 65))
    log_transition[0, 0] = math.log1p(-BIRTH_PROBABILITY)
    log_transition[0, 1:] = math.log(BIRTH_PROBABILITY)+birth_logs
    for i in range(1, 65):
        log_transition[i, 0] = math.log(DEATH_PROBABILITY)
        log_transition[i, 1:] = math.log1p(-DEATH_PROBABILITY) + _positive_grid_logs(
            edges, float(values[i]), PROCESS_SD)
    log_initial = np.concatenate(([math.log(INITIAL_ATOM_PROBABILITY)],
        math.log1p(-INITIAL_ATOM_PROBABILITY)+birth_logs))
    result = dict(values=values, transition=np.exp(log_transition),
        logtransition=log_transition, initial=np.exp(log_initial),
        loginitial=log_initial, positive_bin_edges=edges,
        birth_probabilities=np.exp(birth_logs), birth_log_probabilities=birth_logs)
    _DEFAULT_MODEL_CACHE = {name: value.copy() for name, value in result.items()}
    return result


@njit(cache=False, fastmath=False)
def _cavi_score(q, logq, logprior, values, G, h, means):
    m, k = q.shape
    score = 0.
    for e in range(m):
        score += h[e]*means[e]
        variance = 0.
        for j in range(k):
            variance += q[e, j]*(values[j]-means[e])**2
            if q[e, j] > 0.:
                score -= q[e, j]*(logq[e, j]-logprior[e, j])
        score -= .5*G[e, e]*variance
        for f in range(m):
            score -= .5*means[e]*G[e, f]*means[f]
    return score


@njit(cache=False, fastmath=False)
def _cavi(logprior, prior_q, values, G, h):
    m, k = prior_q.shape
    logq, q = logprior.copy(), prior_q.copy()
    means = np.zeros(m)
    for e in range(m):
        for j in range(k):
            means[e] += q[e, j]*values[j]
    residual = h.copy()
    for e in range(m):
        for f in range(m):
            residual[e] -= G[e, f]*means[f]
    sites = np.empty_like(q)
    cavity = np.empty(m)
    history = np.empty(MAX_SWEEPS+1)
    history[0] = _cavi_score(q, logq, logprior, values, G, h, means)
    converged = False
    max_mean_delta = max_q_delta = 0.
    last_start_mean, last_start_q = means.copy(), q.copy()
    bad_elbo = False
    for sweep in range(MAX_SWEEPS):
        last_start_mean, last_start_q = means.copy(), q.copy()
        for e in range(m):
            cavity[e] = residual[e]+G[e, e]*means[e]
            maximum = -np.inf
            for j in range(k):
                sites[e, j] = -.5*G[e, e]*values[j]**2+cavity[e]*values[j]
                logq[e, j] = logprior[e, j]+sites[e, j]
                maximum = max(maximum, logq[e, j])
            total = 0.
            for j in range(k):
                total += math.exp(logq[e, j]-maximum)
            log_total = math.log(total)
            correction = 0.
            for j in range(k):
                logq[e, j] = (logq[e, j]-maximum)-log_total
                q[e, j] = math.exp(logq[e, j])
                correction += q[e, j]
            new_mean = 0.
            for j in range(k):
                q[e, j] /= correction
                logq[e, j] -= math.log(correction)
                new_mean += q[e, j]*values[j]
            difference = new_mean-means[e]
            means[e] = new_mean
            for f in range(m):
                residual[f] -= G[f, e]*difference
        score = _cavi_score(q, logq, logprior, values, G, h, means)
        history[sweep+1] = score
        if score < history[sweep]-1e-9*max(1., abs(score), abs(history[sweep])):
            bad_elbo = True
        max_mean_delta = 0.
        max_q_delta = 0.
        for e in range(m):
            max_mean_delta = max(max_mean_delta, abs(means[e]-last_start_mean[e]))
            for j in range(k):
                max_q_delta = max(max_q_delta, abs(q[e, j]-last_start_q[e, j]))
        if max_mean_delta <= TOLERANCE and max_q_delta <= TOLERANCE:
            converged = True
            break
    return (logq, q, means, sites, cavity, last_start_mean, last_start_q,
            history[:sweep+2].copy(), sweep+1, converged,
            max_mean_delta, max_q_delta, bad_elbo)


def signature_inventory():
    return dict(cavi=[str(s) for s in _cavi.signatures],
                score=[str(s) for s in _cavi_score.signatures])


def setup():
    """One deterministic artificial m3/K3 warmup; no science stream read."""
    started = time.perf_counter()
    values = np.asarray([0., .3, .7])
    prior = np.full((3, 3), 1./3.)
    result = _cavi(np.log(prior), prior, values,
                  np.diag(np.asarray([2., 4., 6.])), np.asarray([.1, .8, 1.1]))
    if not result[9] or result[-1]:
        raise FloatingPointError('Artificial compiled CAVI warmup failed')
    return dict(artificial_only=True, m=3, K=3, RNG_used=False,
        cache=False, fastmath=False, parallel=False,
        wall_seconds=float(time.perf_counter()-started), signatures=signature_inventory())


def _normalize_logs(logs):
    if np.any(np.isnan(logs)) or np.any(logs == np.inf):
        raise FloatingPointError('Invalid finite-or-negative-infinity log weights')
    normalizer = logsumexp(logs, axis=-1, keepdims=True)
    if not np.all(np.isfinite(normalizer)):
        raise FloatingPointError('Every edge needs reachable posterior support')
    result = logs-normalizer
    probabilities = np.exp(result)
    # This is the ordinary normalization of represented probability mass, not
    # a floor, jitter or replacement of finite logarithmic tail probabilities.
    correction = np.sum(probabilities, axis=-1, keepdims=True)
    probabilities = probabilities/correction
    result = result-np.log(correction)
    return result, probabilities


def _variance(q, values, means):
    result = np.sum(q*(values[None, :]-means[:, None])**2, axis=1)
    _finite(result, 'variance')
    if np.any(result < 0.):
        raise FloatingPointError('Negative centered atomic variance')
    return result


def elbo(q, logq, logprior, values, G, h):
    means = q @ values
    variance = _variance(q, values, means)
    mask = q > 0.
    if np.any(~np.isfinite(logprior[mask])):
        raise FloatingPointError('Posterior puts mass outside reachable prior')
    divergence = float(np.sum(q[mask]*(logq[mask]-logprior[mask])))
    answer = float(-.5*means @ G @ means
        -.5*np.diag(G) @ variance + h @ means-divergence)
    _finite(answer, 'ELBO')
    return answer


def temporal_moments(logold, logtransition, logprior, q, values):
    """Moments of each declared edge-wise old/current variational joint.

    q_current[j] p_old[i] T[i,j] / p_predicted[j] has current marginal q.
    This is the edge-wise variational joint; cross-edge covariance is omitted.
    Centered sums avoid silently clipping cancellation-induced variances.
    """
    m, k = q.shape
    current_mean = q @ values
    current_var = _variance(q, values, current_mean)
    old_mean = np.empty(m)
    old_var = np.empty(m)
    cross = np.empty(m)
    delta_mean = np.empty(m)
    delta_var = np.empty(m)
    current_marginal_error = 0.
    conditional_normalization_error = 0.
    formula_error = 0.
    covariance_determinant_minimum = np.inf
    for e in range(m):
        reachable = np.isfinite(logprior[e])
        backward = np.zeros((k, k))
        logs = logold[e, :, None]+logtransition[:, reachable]
        logs = logs-logprior[e, reachable][None, :]
        conditional = np.exp(logs)
        column_sums = conditional.sum(axis=0)
        if np.any(column_sums <= 0.) or not np.all(np.isfinite(column_sums)):
            raise FloatingPointError('Unresolved reachable backward conditional')
        conditional_normalization_error = max(conditional_normalization_error,
            float(np.max(np.abs(column_sums-1.))))
        conditional /= column_sums[None, :]
        backward[:, reachable] = conditional
        joint = backward*q[e, None, :]
        if np.any(q[e, ~reachable] != 0.):
            raise FloatingPointError('Unreachable current node acquired mass')
        current_marginal_error = max(current_marginal_error,
            float(np.max(np.abs(joint.sum(axis=0)-q[e]))))
        old_marginal = joint.sum(axis=1)
        old_mean[e] = old_marginal @ values
        old_centered = values-old_mean[e]
        current_centered = values-current_mean[e]
        old_var[e] = np.sum(old_marginal*old_centered**2)
        cross[e] = np.sum(joint*old_centered[:, None]*current_centered[None, :])
        delta_mean[e] = current_mean[e]-old_mean[e]
        centered_delta = values[None, :]-values[:, None]-delta_mean[e]
        delta_var[e] = np.sum(joint*centered_delta**2)
        formula = current_var[e]+old_var[e]-2.*cross[e]
        formula_error = max(formula_error, abs(float(formula-delta_var[e])))
        determinant = current_var[e]*old_var[e]-cross[e]**2
        covariance_determinant_minimum = min(covariance_determinant_minimum,
            float(determinant))
    for name, value in [('old mean', old_mean), ('old variance', old_var),
        ('cross covariance', cross), ('delta mean', delta_mean), ('delta variance', delta_var)]:
        _finite(value, name)
    if (np.any(old_var < 0.) or np.any(delta_var < 0.)
            or covariance_determinant_minimum < -1e-12
            or current_marginal_error > 1e-12 or formula_error > 1e-10):
        raise FloatingPointError('Edge-wise temporal joint moment qualification failed')
    return dict(old_smoothed_mean=old_mean, old_smoothed_variance=old_var,
        cross_diag=cross, delta_mean=delta_mean, delta_variance=delta_var,
        temporal_current_marginal_error=current_marginal_error,
        temporal_conditional_normalization_error=conditional_normalization_error,
        temporal_delta_formula_error=formula_error,
        temporal_covariance_determinant_minimum=covariance_determinant_minimum)


class Actor:
    def __init__(self, m, values=None, transition=None, initial=None):
        if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
            raise ValueError('Positive integer full candidate-edge dimension required')
        self.m = int(m)
        if values is None:
            if transition is not None or initial is not None:
                raise ValueError('A custom transition/initial requires custom values')
            self._model = default_model()
        else:
            values = np.asarray(values, dtype=float)
            if (values.ndim != 1 or len(values) < 2 or values[0] != 0.
                    or np.any(np.diff(values) <= 0.) or values[-1] > 1.):
                raise ValueError('Custom values must start at zero and strictly increase to <=1')
            _finite(values, 'custom values')
            transition, initial = np.asarray(transition, dtype=float), np.asarray(initial, dtype=float)
            k = len(values)
            if transition.shape != (k, k) or initial.shape != (k,):
                raise ValueError('Custom Markov and initial dimensions differ')
            _finite(transition, 'custom transition'); _finite(initial, 'custom initial')
            if (np.any(transition < 0.) or np.any(initial < 0.)
                    or np.max(np.abs(transition.sum(axis=1)-1.)) > 1e-12
                    or abs(float(initial.sum()-1.)) > 1e-12):
                raise ValueError('Nonnegative stochastic custom model required')
            log_transition = np.full_like(transition, -np.inf)
            log_initial = np.full_like(initial, -np.inf)
            log_transition[transition > 0.] = np.log(transition[transition > 0.])
            log_initial[initial > 0.] = np.log(initial[initial > 0.])
            self._model = dict(values=values.copy(), transition=transition.copy(),
                logtransition=log_transition, initial=initial.copy(), loginitial=log_initial,
                positive_bin_edges=None, birth_probabilities=None, birth_log_probabilities=None)
        self.values = self._model['values']
        self.k = len(self.values)
        self.transition = self._model['transition']
        self.logtransition = self._model['logtransition']
        self.logq, self.q = _normalize_logs(np.tile(self._model['loginitial'], (self.m, 1)))
        self.step = 0
        self.last = None

    def model_receipt(self):
        return {name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in self._model.items()}

    def update(self, G, h):
        started = time.perf_counter()
        G, h = np.asarray(G, dtype=float), np.asarray(h, dtype=float)
        if G.shape != (self.m, self.m) or h.shape != (self.m,):
            raise ValueError('Full edge likelihood statistic dimensions differ')
        _finite(G, 'G'); _finite(h, 'h')
        symmetry = float(np.max(np.abs(G-G.T)))
        if symmetry > 1e-10*max(1., float(np.max(np.abs(G)))):
            raise ValueError('Symmetric Gram matrix required')
        G = .5*(G+G.T)
        diagonal = np.diag(G)
        if np.any(diagonal < -1e-12):
            raise ValueError('Negative observation Gram diagonal')
        initial = self.step == 0
        old_q = None if initial else self.q.copy()
        logold = None if initial else self.logq.copy()
        if initial:
            logprior, prior_q = self.logq.copy(), self.q.copy()
        else:
            logprior = logsumexp(logold[:, :, None]+self.logtransition[None, :, :], axis=1)
            logprior, prior_q = _normalize_logs(logprior)
        (logq, q, means, sites, cavity, last_start_mean, last_start_q,
         history, sweeps, converged, max_mean_delta, max_q_delta, bad_elbo) = _cavi(
             np.ascontiguousarray(logprior), np.ascontiguousarray(prior_q),
             np.ascontiguousarray(self.values), np.ascontiguousarray(G), np.ascontiguousarray(h))
        if bad_elbo:
            raise FloatingPointError('Coordinate variational ELBO decreased beyond tolerance')
        _finite(q, 'current probabilities'); _finite(means, 'current mean')
        _finite(history, 'ELBO history'); _finite(sites, 'coordinate sites')
        if (np.any(np.isnan(logq)) or np.any(logq == np.inf)
                or np.any(q < 0.) or np.max(np.abs(q.sum(axis=1)-1.)) > 1e-12):
            raise FloatingPointError('Invalid current atomic-node probability distribution')
        means = q @ self.values
        variance = _variance(q, self.values, means)
        positive_mean = []
        support_action = np.empty(self.m)
        for e in range(self.m):
            logmass = float(logsumexp(logq[e, 1:]))
            if np.isfinite(logmass):
                conditional = np.exp(logq[e, 1:]-logmass)
                conditional /= float(conditional.sum())
                value = float(conditional @ self.values[1:])
                positive_mean.append(value)
            else:
                value = None
                positive_mean.append(None)
            if q[e, 0] >= .5:
                support_action[e] = 0.
            elif value is None:
                raise FloatingPointError('Chosen positive support has no reachable positive node')
            else:
                support_action[e] = value
        if (not np.all(np.isfinite(support_action)) or np.any(support_action < 0.)
                or np.any(support_action > 1.) or np.any(means < 0.) or np.any(means > 1.)):
            raise FloatingPointError('Atomic posterior produced illegal graph weights')
        if initial:
            temporal = {name: None for name in ('old_smoothed_mean', 'old_smoothed_variance',
                'cross_diag', 'delta_mean', 'delta_variance', 'temporal_current_marginal_error',
                'temporal_conditional_normalization_error', 'temporal_delta_formula_error',
                'temporal_covariance_determinant_minimum')}
        else:
            temporal = temporal_moments(logold, self.logtransition, logprior, q, self.values)
        result = dict(q=q.copy(), logq=logq.copy(), prior_q=prior_q.copy(),
            logprior=logprior.copy(), old_q=old_q, logold=logold, site=sites.copy(),
            G=G.copy(), h=h.copy(), mean_action=means.copy(),
            support_action=support_action.copy(), variance=variance,
            positive_conditional_mean=positive_mean,
            last_sweep_start_mean=last_start_mean,
            last_sweep_start_q=last_start_q, cavity_scalar=cavity.copy(),
            sweeps=int(sweeps), converged=bool(converged), qualification_passed=bool(converged),
            max_mean_delta=max_mean_delta, max_q_delta=max_q_delta,
            convergence_rule='max_mean_and_max_probability_delta_le_1e-8',
            max_sweeps=MAX_SWEEPS, convergence_tolerance=TOLERANCE,
            elbo_history=history.tolist(), initial=initial, step=self.step,
            primary_action=PRIMARY_ACTION, secondary_action=SECONDARY_ACTION,
            finite_log_weight_underflow_count=int(np.count_nonzero(np.isfinite(logq) & (q == 0.))),
            log_transition_underflow_count=int(np.count_nonzero(
                np.isfinite(self.logtransition) & (self.transition == 0.))),
            exact_unreachable_node_count=int(np.count_nonzero(~np.isfinite(logq))),
            probability_sum_error=float(np.max(np.abs(q.sum(axis=1)-1.))),
            minimum_variance=float(np.min(variance)),
            Gram_symmetry_error=symmetry, posterior_cross_edge_covariance_omitted=True,
            independent_mean_field_edges=True, exact_continuous_posterior_claimed=False,
            model_grid_min_positive=float(self.values[1]), physical_positive_gap_assumed=False,
            old_filtered_mean=None if initial else old_q @ self.values,
            actor_seconds=float(time.perf_counter()-started))
        result.update(temporal)
        self.q, self.logq = q.copy(), logq.copy()
        self.step += 1
        self.last = result
        return result

"""Persistent/reset amplitude process, same-kernel dense and O(mK) controls.

Definitions only: numerical imports, the frozen V46 module, and compilation are
lazy.  This changes the positive amplitude process; it is not an optimization
equivalent to the old Gaussian transition.  All states/log tails are retained.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OLD_SOURCE = PROJECT/'continuation_20261025_atomic_nodes_v46/atomic_nodes.py'
OLD_SHA256 = 'cccedcdc135d8ee4ca4b088ceca66b3f347d762af57e785cd13843d70659a30f'
BIRTH_PROBABILITY = .01
DEATH_PROBABILITY = .01
RESET_PROBABILITY = .01
MAX_SWEEPS = 30
TOLERANCE = 1e-8
KERNEL_NAME = 'v60_positive_persistent_reset_diag_rank1'
STATE_VERSION = 'v60_bounded_actor_state_1'
BRANCH_NAMES = ('zero_zero', 'birth', 'death', 'persistent', 'reset')
PRIMARY_ACTION = 'factorized_support_MAP_conditional_mean'
SECONDARY_ACTION = 'posterior_mean'
_OLD = None
_FAST_KERNELS = None

TEMPORAL_ARRAY_KEYS = ('old_smoothed_q', 'old_smoothed_logq', 'old_smoothed_mean',
    'old_smoothed_second_moment', 'old_smoothed_variance', 'old_smoothed_cov',
    'current_second_moment', 'cross_raw_moment', 'cross_diag', 'cross_cov',
    'delta_mean', 'delta_second_moment', 'delta_variance', 'delta_cov',
    'branch_weights', 'branch_log_weights')
TEMPORAL_DIAGNOSTIC_KEYS = ('temporal_current_marginal_error',
    'temporal_old_marginal_error', 'temporal_conditional_normalization_error',
    'temporal_delta_formula_error', 'temporal_covariance_determinant_minimum',
    'temporal_pair_normalization_error', 'temporal_raw_centered_identity_error',
    'temporal_finite_log_underflow_count')


def _numeric():
    import numpy as np
    from scipy.special import logsumexp
    return np, logsumexp


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_old():
    global _OLD
    if _OLD is None:
        if _sha(OLD_SOURCE) != OLD_SHA256:
            raise ImportError('Frozen original V46 CAVI source changed')
        alias = '_v60_frozen_v46_cavi'
        if alias in sys.modules:
            candidate = sys.modules[alias]
            if Path(candidate.__file__).resolve() != OLD_SOURCE.resolve():
                raise ImportError('Private V46 CAVI alias has a different origin')
        else:
            spec = importlib.util.spec_from_file_location(alias, OLD_SOURCE)
            candidate = importlib.util.module_from_spec(spec)
            sys.modules[alias] = candidate
            spec.loader.exec_module(candidate)
        if candidate.MAX_SWEEPS != MAX_SWEEPS or candidate.TOLERANCE != TOLERANCE:
            raise ImportError('Frozen original finite CAVI constants differ')
        _OLD = candidate
    return _OLD


def _normalize(logs):
    np, lse = _numeric()
    logs = np.asarray(logs, dtype=float)
    if np.any(np.isnan(logs)) or np.any(logs == np.inf):
        raise FloatingPointError('Only finite or -inf log weights are legal')
    normalizer = lse(logs, axis=-1, keepdims=True)
    if not np.all(np.isfinite(normalizer)):
        raise FloatingPointError('Empty reachable distribution')
    result = logs-normalizer
    q = np.exp(result)
    correction = q.sum(axis=-1, keepdims=True)
    q = q/correction
    result = result-np.log(correction)
    return result, q


def _log_transition(np, logr):
    k = len(logr)+1
    b, d, rho = BIRTH_PROBABILITY, DEATH_PROBABILITY, RESET_PROBABILITY
    answer = np.empty((k, k), dtype=float)
    answer[0, 0] = math.log1p(-b)
    answer[0, 1:] = math.log(b)+logr
    answer[1:, 0] = math.log(d)
    answer[1:, 1:] = math.log1p(-d)+math.log(rho)+logr[None, :]
    for j in range(k-1):
        answer[j+1, j+1] = math.log1p(-d)+float(np.logaddexp(
            math.log1p(-rho), math.log(rho)+logr[j]))
    return answer


def make_model(original):
    """Copy original data; replace positive process with exactly equation (1).

    Original finite birth logs, initial logs and values are copied directly.
    Artificial smaller grids are allowed for qualification; production must
    separately admit canonical_grid65=True.  No new process parameter is exposed.
    """
    np, lse = _numeric()
    model = {key: copy.deepcopy(value) for key, value in original.items() if value is not None}
    omitted = sorted(key for key, value in original.items() if value is None)
    values = np.array(original['values'], dtype=float, order='C', copy=True)
    k = len(values)
    if (values.ndim != 1 or k < 2 or values[0] != 0. or values[-1] > 1.
            or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.)):
        raise ValueError('Finite increasing nonnegative grid with zero first required')
    if original.get('birth_log_probabilities') is None:
        raise ValueError('Original birth log probabilities required without exp-log reconstruction')
    logr = np.array(original['birth_log_probabilities'], dtype=float, order='C', copy=True)
    loginitial = np.array(original['loginitial'], dtype=float, order='C', copy=True)
    if (logr.shape != (k-1,) or not np.all(np.isfinite(logr))
            or abs(float(lse(logr))) > 1e-12 or loginitial.shape != (k,)
            or np.any(np.isnan(loginitial)) or np.any(loginitial == np.inf)
            or abs(float(lse(loginitial))) > 1e-12):
        raise ValueError('Normalized original birth/initial log masses required')
    if 'logtransition' in original:
        model['original_logtransition'] = np.array(original['logtransition'], dtype=float, copy=True)
    if 'transition' in original:
        model['original_transition'] = np.array(original['transition'], dtype=float, copy=True)
    logT = _log_transition(np, logr)
    alpha = (1.-DEATH_PROBABILITY)*(1.-RESET_PROBABILITY)
    beta = (1.-DEATH_PROBABILITY)*RESET_PROBABILITY
    lower_logs = np.r_[math.log(min(1.-BIRTH_PROBABILITY, DEATH_PROBABILITY)),
        math.log(min(BIRTH_PROBABILITY, beta))+logr]
    model.update(values=values, loginitial=loginitial,
        initial=np.array(original.get('initial', np.exp(loginitial)), dtype=float, copy=True),
        birth_log_probabilities=logr,
        birth_probabilities=np.exp(logr), logtransition=logT, transition=np.exp(logT),
        birth_probability=BIRTH_PROBABILITY, death_probability=DEATH_PROBABILITY,
        reset_probability=RESET_PROBABILITY, alpha=alpha, beta=beta,
        logalpha=math.log1p(-DEATH_PROBABILITY)+math.log1p(-RESET_PROBABILITY),
        logbeta=math.log1p(-DEATH_PROBABILITY)+math.log(RESET_PROBABILITY),
        log_prediction_lower_bounds=lower_logs, prediction_lower_bounds=np.exp(lower_logs),
        kernel_name=KERNEL_NAME, amplitude_process_changed=True,
        original_gaussian_transition_equivalence_claimed=False,
        original_none_keys_omitted=','.join(omitted),
        canonical_grid65=bool(k == 65 and np.array_equal(values, np.arange(65, dtype=float)/64.)))
    if np.max(np.abs(lse(logT, axis=1))) > 1e-12:
        raise FloatingPointError('New transition rows are not normalized')
    return model


def _model_identity(model):
    np, unused = _numeric()
    digest = hashlib.sha256()
    keys = ('values', 'loginitial', 'birth_log_probabilities', 'logtransition', 'transition',
            'birth_probability', 'death_probability', 'reset_probability',
            'alpha', 'beta', 'logalpha', 'logbeta', 'log_prediction_lower_bounds')
    for key in keys:
        value = np.ascontiguousarray(model[key])
        if value.dtype.hasobject:
            raise ValueError('Object model identity is forbidden')
        digest.update(key.encode('ascii')+b'\0')
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_model(model):
    np, lse = _numeric()
    values = np.asarray(model['values'], dtype=float)
    k = len(values)
    logr = np.asarray(model['birth_log_probabilities'], dtype=float)
    logT = np.asarray(model['logtransition'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    initial = np.asarray(model['loginitial'], dtype=float)
    expected_alpha = (1.-DEATH_PROBABILITY)*(1.-RESET_PROBABILITY)
    expected_beta = (1.-DEATH_PROBABILITY)*RESET_PROBABILITY
    expected_logalpha = math.log1p(-DEATH_PROBABILITY)+math.log1p(-RESET_PROBABILITY)
    expected_logbeta = math.log1p(-DEATH_PROBABILITY)+math.log(RESET_PROBABILITY)
    if (model.get('kernel_name') != KERNEL_NAME or model.get('birth_probability') != BIRTH_PROBABILITY
            or model.get('death_probability') != DEATH_PROBABILITY
            or model.get('reset_probability') != RESET_PROBABILITY
            or model.get('alpha') != expected_alpha or model.get('beta') != expected_beta
            or model.get('logalpha') != expected_logalpha or model.get('logbeta') != expected_logbeta):
        raise ValueError('The one fixed V60 process model is required')
    if (values.ndim != 1 or k < 2 or values[0] != 0. or values[-1] > 1.
            or np.any(np.diff(values) <= 0.) or not np.all(np.isfinite(values))
            or logr.shape != (k-1,) or not np.all(np.isfinite(logr))
            or logT.shape != (k, k) or not np.all(np.isfinite(logT))
            or transition.shape != (k, k) or not np.array_equal(transition, np.exp(logT))
            or initial.shape != (k,) or np.any(np.isnan(initial)) or np.any(initial == np.inf)
            or abs(float(lse(initial))) > 1e-12 or abs(float(lse(logr))) > 1e-12
            or not np.array_equal(logT, _log_transition(np, logr))):
        raise ValueError('Invalid or inconsistent fixed V60 model')


def predict_dense(logold, model):
    """Genuine all-state K-by-K log-domain Markov sum."""
    np, lse = _numeric()
    old = np.asarray(logold, dtype=float)
    if old.ndim != 2 or old.shape[1] != len(model['values']):
        raise ValueError('Edge-by-state old logs required')
    raw = lse(old[:, :, None]+np.asarray(model['logtransition'])[None, :, :], axis=1)
    return _normalize(raw)


def predict_fast_reference(logold, model):
    """Auditable NumPy reference for the O(mK) prediction formula."""
    np, lse = _numeric()
    old = np.asarray(logold, dtype=float)
    if old.ndim != 2 or old.shape[1] != len(model['values']):
        raise ValueError('Edge-by-state old logs required')
    positive = lse(old[:, 1:], axis=1)
    reset_birth = np.logaddexp(math.log(BIRTH_PROBABILITY)+old[:, 0],
        model['logbeta']+positive)
    raw = np.empty_like(old)
    raw[:, 0] = np.logaddexp(math.log1p(-BIRTH_PROBABILITY)+old[:, 0],
        math.log(DEATH_PROBABILITY)+positive)
    raw[:, 1:] = np.logaddexp(model['logalpha']+old[:, 1:],
        reset_birth[:, None]+np.asarray(model['birth_log_probabilities'])[None, :])
    return _normalize(raw)


def _empty_temporal():
    return {key: None for key in TEMPORAL_ARRAY_KEYS+TEMPORAL_DIAGNOSTIC_KEYS}


def _moment_result(q, old_logs, old_q, cross, raw_cross, delta_var, delta_second,
                   branch_logs, branch_q, values, current_error, conditional_error,
                   pair_error):
    np, unused = _numeric()
    current_mean, old_mean = q @ values, old_q @ values
    current_var = np.sum(q*(values[None, :]-current_mean[:, None])**2, axis=1)
    old_var = np.sum(old_q*(values[None, :]-old_mean[:, None])**2, axis=1)
    current_second = np.sum(q*values[None, :]**2, axis=1)
    old_second = np.sum(old_q*values[None, :]**2, axis=1)
    delta = current_mean-old_mean
    formula_error = float(np.max(np.abs(delta_var-(current_var+old_var-2.*cross))))
    identity_error = float(max(np.max(np.abs(old_second-old_var-old_mean**2)),
        np.max(np.abs(current_second-current_var-current_mean**2)),
        np.max(np.abs(delta_second-delta_var-delta**2)),
        np.max(np.abs(raw_cross-cross-old_mean*current_mean))))
    determinant = float(np.min(current_var*old_var-cross**2))
    arrays = (old_q, old_mean, old_var, cross, raw_cross, delta, delta_var, delta_second, branch_q)
    if (not all(np.all(np.isfinite(value)) for value in arrays)
            or np.any(old_q < 0.) or np.any(old_var < 0.) or np.any(delta_var < 0.)
            or current_error > 1e-12 or formula_error > 1e-10 or identity_error > 1e-10
            or determinant < -1e-12 or pair_error > 1e-12):
        raise FloatingPointError('Same-information V60 temporal moment qualification failed')
    return dict(old_smoothed_q=old_q, old_smoothed_logq=old_logs,
        old_smoothed_mean=old_mean, old_smoothed_second_moment=old_second,
        old_smoothed_variance=old_var, old_smoothed_cov=np.diag(old_var),
        current_second_moment=current_second, cross_raw_moment=raw_cross,
        cross_diag=cross, cross_cov=np.diag(cross), delta_mean=delta,
        delta_second_moment=delta_second, delta_variance=delta_var, delta_cov=np.diag(delta_var),
        branch_weights=branch_q, branch_log_weights=branch_logs,
        temporal_current_marginal_error=float(current_error), temporal_old_marginal_error=0.,
        temporal_conditional_normalization_error=float(conditional_error),
        temporal_delta_formula_error=formula_error,
        temporal_covariance_determinant_minimum=determinant,
        temporal_pair_normalization_error=float(pair_error),
        temporal_raw_centered_identity_error=identity_error,
        temporal_finite_log_underflow_count=int(np.count_nonzero(np.isfinite(old_logs) & (old_q == 0.))))


def temporal_dense(logold, logprior, logq, q, model):
    """Independent full dense pair sums, no structural moment shortcut."""
    np, lse = _numeric()
    p, prior, current_logs, q = (np.asarray(x, dtype=float) for x in (logold, logprior, logq, q))
    m, k = q.shape
    values, logT = np.asarray(model['values']), np.asarray(model['logtransition'])
    logr = np.asarray(model['birth_log_probabilities'])
    old_logs, old_q = np.empty_like(q), np.empty_like(q)
    cross, raw_cross, delta_var, delta_second = (np.empty(m) for unused in range(4))
    branch_logs, branch_q = np.empty((m, 5)), np.empty((m, 5))
    current_error = conditional_error = pair_error = 0.
    for e in range(m):
        conditional = p[e, :, None]+logT-prior[e, None, :]
        conditional_error = max(conditional_error,
            float(np.max(np.abs(np.exp(lse(conditional, axis=0))-1.))))
        raw = conditional+current_logs[e, None, :]
        pair_logs, joint = _normalize(raw.reshape(-1))
        pair_logs, joint = pair_logs.reshape(k, k), joint.reshape(k, k)
        old_logs[e], old_q[e] = _normalize(lse(pair_logs, axis=1))
        current_error = max(current_error, float(np.max(np.abs(joint.sum(axis=0)-q[e]))))
        pair_error = max(pair_error, abs(float(joint.sum())-1.))
        old_mean, current_mean = float(old_q[e] @ values), float(q[e] @ values)
        old_centered, current_centered = values-old_mean, values-current_mean
        cross[e] = float(np.sum(joint*old_centered[:, None]*current_centered[None, :]))
        raw_cross[e] = float(np.sum(joint*values[:, None]*values[None, :]))
        difference = values[None, :]-values[:, None]
        delta_mean = current_mean-old_mean
        delta_var[e] = float(np.sum(joint*(difference-delta_mean)**2))
        delta_second[e] = float(np.sum(joint*difference**2))
        c = current_logs[e]-prior[e]
        # Dense reference explicitly sums the positive-positive reset matrix.
        branch_raw = np.asarray([
            p[e, 0]+math.log1p(-BIRTH_PROBABILITY)+c[0],
            float(lse(p[e, 0]+math.log(BIRTH_PROBABILITY)+logr+c[1:])),
            float(lse(p[e, 1:]+math.log(DEATH_PROBABILITY)+c[0])),
            float(lse(model['logalpha']+p[e, 1:]+c[1:])),
            float(lse(p[e, 1:, None]+model['logbeta']+logr[None, :]+c[None, 1:]))])
        branch_logs[e], branch_q[e] = _normalize(branch_raw)
    return _moment_result(q, old_logs, old_q, cross, raw_cross, delta_var,
        delta_second, branch_logs, branch_q, values, current_error, conditional_error, pair_error)


def _conditional(logs):
    np, lse = _numeric()
    total = float(lse(logs))
    if total == -np.inf:
        return np.full_like(logs, -np.inf), np.zeros_like(logs), total
    normalized, q = _normalize(logs)
    return normalized, q, total


def temporal_fast_reference(logold, logprior, logq, q, model):
    """Auditable NumPy five-branch reference, not the production backend."""
    np, lse = _numeric()
    p, prior, current_logs, q = (np.asarray(x, dtype=float) for x in (logold, logprior, logq, q))
    m, k = q.shape
    values = np.asarray(model['values'])
    positive_values = values[1:]
    logr = np.asarray(model['birth_log_probabilities'])
    old_logs, old_q = np.empty_like(q), np.empty_like(q)
    cross, raw_cross, delta_var, delta_second = (np.empty(m) for unused in range(4))
    branch_logs, branch_q = np.empty((m, 5)), np.empty((m, 5))
    current_error = conditional_error = pair_error = 0.
    for e in range(m):
        c = current_logs[e]-prior[e]
        loga, a, logs = _conditional(p[e, 1:])
        logh, h, logR = _conditional(logr+c[1:])
        logg, g, logP = _conditional(p[e, 1:]+c[1:])
        branch_raw = np.asarray([
            p[e, 0]+math.log1p(-BIRTH_PROBABILITY)+c[0],
            p[e, 0]+math.log(BIRTH_PROBABILITY)+logR,
            logs+math.log(DEATH_PROBABILITY)+c[0],
            model['logalpha']+logP, model['logbeta']+logs+logR])
        branch_logs[e], branch_q[e] = _normalize(branch_raw)
        lw, w = branch_logs[e], branch_q[e]
        old_raw = np.empty(k)
        old_raw[0] = np.logaddexp(lw[0], lw[1])
        old_raw[1:] = lse(np.vstack((lw[2]+loga, lw[4]+loga, lw[3]+logg)), axis=0)
        old_logs[e], old_q[e] = _normalize(old_raw)
        current_raw = np.empty(k)
        current_raw[0] = np.logaddexp(lw[0], lw[2])
        current_raw[1:] = lse(np.vstack((lw[1]+logh, lw[4]+logh, lw[3]+logg)), axis=0)
        unused_logs, reconstructed_current = _normalize(current_raw)
        current_error = max(current_error, float(np.max(np.abs(reconstructed_current-q[e]))))
        pair_error = max(pair_error, abs(float(w.sum())-1.))
        # Recompute normalization of reverse columns in O(K), without ratios.
        inflow = np.logaddexp(math.log(BIRTH_PROBABILITY)+p[e, 0], model['logbeta']+logs)
        col0 = np.logaddexp(math.log1p(-BIRTH_PROBABILITY)+p[e, 0], math.log(DEATH_PROBABILITY)+logs)
        colp = np.logaddexp(model['logalpha']+p[e, 1:], inflow+logr)
        conditional_error = max(conditional_error, abs(float(np.exp(col0-prior[e, 0]))-1.),
            float(np.max(np.abs(np.exp(colp-prior[e, 1:])-1.))))
        ma, mh, mg = float(a @ positive_values), float(h @ positive_values), float(g @ positive_values)
        va = float(np.sum(a*(positive_values-ma)**2))
        vh = float(np.sum(h*(positive_values-mh)**2))
        vg = float(np.sum(g*(positive_values-mg)**2))
        mo, mc = float(old_q[e] @ values), float(q[e] @ values)
        delta = mc-mo
        cross[e] = (w[0]*mo*mc-w[1]*mo*(mh-mc)-w[2]*(ma-mo)*mc
            +w[3]*(vg+(mg-mo)*(mg-mc))+w[4]*(ma-mo)*(mh-mc))
        raw_cross[e] = w[3]*(vg+mg*mg)+w[4]*ma*mh
        delta_var[e] = ((w[0]+w[3])*delta*delta+w[1]*(vh+(mh-delta)**2)
            +w[2]*(va+(-ma-delta)**2)+w[4]*(va+vh+(mh-ma-delta)**2))
        delta_second[e] = w[1]*(vh+mh*mh)+w[2]*(va+ma*ma)+w[4]*(va+vh+(mh-ma)**2)
    return _moment_result(q, old_logs, old_q, cross, raw_cross, delta_var,
        delta_second, branch_logs, branch_q, values, current_error, conditional_error, pair_error)


def _compiled():
    """Create lazy dispatchers only after the owner's runtime admission.

    No compilation/cache/files are created on module import.  All kernels use
    full finite log tails, ordinary represented-mass normalization, no floor,
    no clipping, no fastmath and no parallel reduction.
    """
    global _FAST_KERNELS
    if _FAST_KERNELS is not None:
        return _FAST_KERNELS
    import numpy as np
    from numba import njit

    @njit(cache=False, fastmath=False)
    def logadd(a, b):
        if a == -np.inf:
            return b
        if b == -np.inf:
            return a
        maximum, minimum = (a, b) if a >= b else (b, a)
        return maximum+math.log1p(math.exp(minimum-maximum))

    @njit(cache=False, fastmath=False)
    def lse1(raw):
        maximum = -np.inf
        for j in range(len(raw)):
            maximum = max(maximum, raw[j])
        if maximum == -np.inf:
            return -np.inf
        total = 0.
        for j in range(len(raw)):
            total += math.exp(raw[j]-maximum)
        return maximum+math.log(total)

    @njit(cache=False, fastmath=False)
    def normalize(raw):
        maximum = -np.inf
        for j in range(len(raw)):
            maximum = max(maximum, raw[j])
        if maximum == -np.inf or not math.isfinite(maximum):
            raise FloatingPointError('Empty or invalid reachable log weights')
        total = 0.
        for j in range(len(raw)):
            total += math.exp(raw[j]-maximum)
        logtotal = math.log(total)
        logs, probability = np.empty_like(raw), np.empty_like(raw)
        correction = 0.
        for j in range(len(raw)):
            logs[j] = (raw[j]-maximum)-logtotal
            probability[j] = math.exp(logs[j])
            correction += probability[j]
        logcorrection = math.log(correction)
        for j in range(len(raw)):
            logs[j] -= logcorrection
            probability[j] /= correction
        return logs, probability

    @njit(cache=False, fastmath=False)
    def conditional(raw):
        total = lse1(raw)
        if total == -np.inf:
            return np.full_like(raw, -np.inf), np.zeros_like(raw), total
        logs, probability = normalize(raw)
        return logs, probability, total

    @njit(cache=False, fastmath=False)
    def predict_kernel(old, logr, logalpha, logbeta):
        m, k = old.shape
        logs, probability = np.empty_like(old), np.empty_like(old)
        raw = np.empty(k)
        for e in range(m):
            positive = lse1(old[e, 1:])
            inflow = logadd(math.log(BIRTH_PROBABILITY)+old[e, 0], logbeta+positive)
            raw[0] = logadd(math.log1p(-BIRTH_PROBABILITY)+old[e, 0],
                math.log(DEATH_PROBABILITY)+positive)
            for j in range(1, k):
                raw[j] = logadd(logalpha+old[e, j], inflow+logr[j-1])
            rowlogs, rowq = normalize(raw)
            logs[e], probability[e] = rowlogs, rowq
        return logs, probability

    @njit(cache=False, fastmath=False)
    def temporal_kernel(p, prior, current_logs, q, values, logr, logalpha, logbeta):
        m, k = q.shape
        old_logs, old_q = np.empty_like(q), np.empty_like(q)
        cross, raw_cross = np.empty(m), np.empty(m)
        delta_var, delta_second = np.empty(m), np.empty(m)
        branch_logs, branch_q = np.empty((m, 5)), np.empty((m, 5))
        current_error = conditional_error = pair_error = 0.
        c, oldraw, currentraw = np.empty(k), np.empty(k), np.empty(k)
        rawa, rawh, rawg = np.empty(k-1), np.empty(k-1), np.empty(k-1)
        branchraw = np.empty(5)
        for e in range(m):
            for j in range(k):
                c[j] = current_logs[e, j]-prior[e, j]
            for j in range(k-1):
                rawa[j] = p[e, j+1]
                rawh[j] = logr[j]+c[j+1]
                rawg[j] = p[e, j+1]+c[j+1]
            loga, a, logs = conditional(rawa)
            logh, h, logR = conditional(rawh)
            logg, g, logP = conditional(rawg)
            branchraw[0] = p[e, 0]+math.log1p(-BIRTH_PROBABILITY)+c[0]
            branchraw[1] = p[e, 0]+math.log(BIRTH_PROBABILITY)+logR
            branchraw[2] = logs+math.log(DEATH_PROBABILITY)+c[0]
            branchraw[3] = logalpha+logP
            branchraw[4] = logbeta+logs+logR
            lw, w = normalize(branchraw)
            branch_logs[e], branch_q[e] = lw, w
            oldraw[0] = logadd(lw[0], lw[1])
            currentraw[0] = logadd(lw[0], lw[2])
            for j in range(k-1):
                oldraw[j+1] = logadd(logadd(lw[2]+loga[j], lw[4]+loga[j]), lw[3]+logg[j])
                currentraw[j+1] = logadd(logadd(lw[1]+logh[j], lw[4]+logh[j]), lw[3]+logg[j])
            rowlogs, rowq = normalize(oldraw)
            old_logs[e], old_q[e] = rowlogs, rowq
            unused_logs, reconstructed_current = normalize(currentraw)
            for j in range(k):
                current_error = max(current_error, abs(reconstructed_current[j]-q[e, j]))
            mass = 0.
            for j in range(5):
                mass += w[j]
            pair_error = max(pair_error, abs(mass-1.))
            inflow = logadd(math.log(BIRTH_PROBABILITY)+p[e, 0], logbeta+logs)
            col0 = logadd(math.log1p(-BIRTH_PROBABILITY)+p[e, 0], math.log(DEATH_PROBABILITY)+logs)
            conditional_error = max(conditional_error, abs(math.exp(col0-prior[e, 0])-1.))
            for j in range(k-1):
                col = logadd(logalpha+p[e, j+1], inflow+logr[j])
                conditional_error = max(conditional_error, abs(math.exp(col-prior[e, j+1])-1.))
            ma = mh = mg = 0.
            for j in range(k-1):
                ma += a[j]*values[j+1]
                mh += h[j]*values[j+1]
                mg += g[j]*values[j+1]
            va = vh = vg = 0.
            for j in range(k-1):
                va += a[j]*(values[j+1]-ma)**2
                vh += h[j]*(values[j+1]-mh)**2
                vg += g[j]*(values[j+1]-mg)**2
            mo = mc = 0.
            for j in range(k):
                mo += rowq[j]*values[j]
                mc += q[e, j]*values[j]
            delta = mc-mo
            cross[e] = (w[0]*mo*mc-w[1]*mo*(mh-mc)-w[2]*(ma-mo)*mc
                +w[3]*(vg+(mg-mo)*(mg-mc))+w[4]*(ma-mo)*(mh-mc))
            raw_cross[e] = w[3]*(vg+mg*mg)+w[4]*ma*mh
            delta_var[e] = ((w[0]+w[3])*delta*delta+w[1]*(vh+(mh-delta)**2)
                +w[2]*(va+(-ma-delta)**2)+w[4]*(va+vh+(mh-ma-delta)**2))
            delta_second[e] = w[1]*(vh+mh*mh)+w[2]*(va+ma*ma)+w[4]*(va+vh+(mh-ma)**2)
        return (old_logs, old_q, cross, raw_cross, delta_var, delta_second,
            branch_logs, branch_q, current_error, conditional_error, pair_error)

    _FAST_KERNELS = dict(predict=predict_kernel, temporal=temporal_kernel,
        normalize=normalize, conditional=conditional, logsumexp=lse1, logaddexp=logadd)
    return _FAST_KERNELS


def predict_fast(logold, model):
    """Production O(mK) compiled prediction; fixed full T remains in model."""
    np, unused = _numeric()
    old = np.ascontiguousarray(logold, dtype=float)
    if (old.ndim != 2 or old.shape[1] != len(model['values'])
            or np.any(np.isnan(old)) or np.any(old == np.inf)):
        raise ValueError('Edge-by-state finite-or--inf old logs required')
    return _compiled()['predict'](old,
        np.ascontiguousarray(model['birth_log_probabilities']), model['logalpha'], model['logbeta'])


def temporal_fast(logold, logprior, logq, q, model):
    """Production O(mK) compiled five-branch temporal calculation."""
    np, unused = _numeric()
    p, prior, current_logs, probability = (
        np.ascontiguousarray(value, dtype=float) for value in (logold, logprior, logq, q))
    if (probability.ndim != 2 or probability.shape[1] != len(model['values'])
            or p.shape != probability.shape or prior.shape != probability.shape
            or current_logs.shape != probability.shape or not np.all(np.isfinite(prior))
            or not np.all(np.isfinite(probability)) or np.any(probability < 0.)):
        raise ValueError('Consistent full edge-by-state temporal arrays required')
    for logs in (p, current_logs):
        if np.any(np.isnan(logs)) or np.any(logs == np.inf):
            raise ValueError('Temporal logs must be finite or -inf')
    values = np.ascontiguousarray(model['values'])
    result = _compiled()['temporal'](p, prior, current_logs, probability, values,
        np.ascontiguousarray(model['birth_log_probabilities']), model['logalpha'], model['logbeta'])
    (old_logs, old_q, cross, raw_cross, delta_var, delta_second,
     branch_logs, branch_q, current_error, conditional_error, pair_error) = result
    return _moment_result(probability, old_logs, old_q, cross, raw_cross, delta_var,
        delta_second, branch_logs, branch_q, values, current_error, conditional_error, pair_error)


def artificial_cavi(logprior, prior_q, values, G, h, max_sweeps=1):
    """Artificial-only cap fixture using the exact frozen Python function body.

    Copied globals isolate the cap: no old module global or compiled production
    dispatcher is modified.  Production Actor.update never calls this helper.
    """
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or not 1 <= max_sweeps <= MAX_SWEEPS:
        raise ValueError('Artificial integer cap in 1..30 required')
    old = _load_old()
    namespace = dict(old._cavi.py_func.__globals__)
    namespace['MAX_SWEEPS'] = max_sweeps
    namespace['_cavi_score'] = old._cavi_score.py_func
    function = types.FunctionType(old._cavi.py_func.__code__, namespace, '_v60_artificial_cavi')
    return function(logprior, prior_q, values, G, h)


class Actor:
    def __init__(self, m, model, mode='fast'):
        np, unused = _numeric()
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError('Positive integer full candidate-edge dimension required')
        if mode not in ('dense', 'fast'):
            raise ValueError('Mode must be dense or fast')
        _validate_model(model)
        self.m, self.mode = int(m), mode
        self._model = copy.deepcopy(model)
        self.values = np.array(model['values'], dtype=float, order='C', copy=True)
        self.k = len(self.values)
        self.transition = np.array(model['transition'], dtype=float, order='C', copy=True)
        self.logtransition = np.array(model['logtransition'], dtype=float, order='C', copy=True)
        self.model_identity = _model_identity(self._model)
        self.logq, self.q = _load_old()._normalize_logs(np.tile(model['loginitial'], (self.m, 1)))
        self.step, self.last = 0, None

    def model_receipt(self):
        return copy.deepcopy(self._model)

    def update(self, G, h):
        np, lse = _numeric()
        started = time.perf_counter()
        old = _load_old()
        G, h = np.array(G, dtype=float, order='C', copy=True), np.array(h, dtype=float, order='C', copy=True)
        if (G.shape != (self.m, self.m) or h.shape != (self.m,)
                or not np.all(np.isfinite(G)) or not np.all(np.isfinite(h))):
            raise ValueError('Finite full edge likelihood statistics required')
        symmetry = float(np.max(np.abs(G-G.T)))
        if symmetry > 1e-10*max(1., float(np.max(np.abs(G)))):
            raise ValueError('Symmetric Gram required')
        G = np.ascontiguousarray(.5*(G+G.T))
        if np.any(np.diag(G) < -1e-12):
            raise ValueError('Negative observation Gram diagonal')
        initial = self.step == 0
        old_q, logold = (None, None) if initial else (self.q.copy(), self.logq.copy())
        clock = time.perf_counter()
        if initial:
            logprior, prior_q = self.logq.copy(), self.q.copy()
        elif self.mode == 'dense':
            logprior, prior_q = predict_dense(logold, self._model)
        else:
            logprior, prior_q = predict_fast(logold, self._model)
        prediction_seconds = time.perf_counter()-clock
        clock = time.perf_counter()
        (logq, q, means, sites, cavity, start_mean, start_q, history, sweeps,
         converged, mean_delta, q_delta, bad_elbo) = old._cavi(
            np.ascontiguousarray(logprior), np.ascontiguousarray(prior_q), self.values,
            G, np.ascontiguousarray(h))
        cavi_seconds = time.perf_counter()-clock
        if bad_elbo:
            raise FloatingPointError('Frozen finite CAVI ELBO decreased beyond its tolerance')
        if (not all(np.all(np.isfinite(x)) for x in (q, means, sites, history))
                or np.any(np.isnan(logq)) or np.any(logq == np.inf) or np.any(q < 0.)
                or np.max(np.abs(q.sum(axis=1)-1.)) > 1e-12):
            raise FloatingPointError('Invalid current full atomic posterior')
        clock = time.perf_counter()
        means = q @ self.values
        variance = np.sum(q*(self.values[None, :]-means[:, None])**2, axis=1)
        support, positive = np.empty(self.m), []
        for e in range(self.m):
            logmass = float(lse(logq[e, 1:]))
            value = None
            if np.isfinite(logmass):
                conditional = np.exp(logq[e, 1:]-logmass)
                conditional /= float(conditional.sum())
                value = float(conditional @ self.values[1:])
            positive.append(value)
            if q[e, 0] >= .5:
                support[e] = 0.
            elif value is None:
                raise FloatingPointError('Positive support selected without a positive node')
            else:
                support[e] = value
        if (not np.all(np.isfinite(support)) or np.any((support < 0.) | (support > 1.))
                or np.any((means < 0.) | (means > 1.)) or np.any(variance < 0.)):
            raise FloatingPointError('Illegal atomic readout')
        readout_seconds = time.perf_counter()-clock
        clock = time.perf_counter()
        if initial:
            temporal = _empty_temporal()
        elif self.mode == 'dense':
            temporal = temporal_dense(logold, logprior, logq, q, self._model)
        else:
            temporal = temporal_fast(logold, logprior, logq, q, self._model)
        temporal_seconds = time.perf_counter()-clock
        result = dict(q=q.copy(), logq=logq.copy(), prior_q=prior_q.copy(), logprior=logprior.copy(),
            current_mean=means.copy(), current_cov=np.diag(variance), mean_action=means.copy(),
            support_action=support.copy(), variance=variance,
            positive_conditional_mean=positive, old_q=old_q, logold=logold,
            old_filtered_mean=None if initial else old_q @ self.values,
            G=G.copy(), h=h.copy(), site=sites.copy(), cavity_scalar=cavity.copy(),
            last_sweep_start_mean=start_mean.copy(), last_sweep_start_q=start_q.copy(),
            sweeps=int(sweeps), iterations=int(sweeps), converged=bool(converged),
            cap=bool(not converged), qualification_passed=bool(converged), bad_elbo=bool(bad_elbo),
            max_mean_delta=float(mean_delta), max_q_delta=float(q_delta),
            convergence_rule='max_mean_and_max_probability_delta_le_1e-8',
            max_sweeps=MAX_SWEEPS, convergence_tolerance=TOLERANCE, elbo_history=history.tolist(),
            initial=initial, step=self.step, mode=self.mode, kernel_name=KERNEL_NAME,
            primary_action=PRIMARY_ACTION, secondary_action=SECONDARY_ACTION,
            model_identity=self.model_identity, cavi_source_sha256=OLD_SHA256,
            cavi_body_reused='original_V46._cavi', last_used_sites_preserved=True,
            final_unrecorded_refresh=False, prediction_performed=bool(not initial),
            prediction_log_lower_bound_min_margin=float(np.min(logprior-np.asarray(self._model['log_prediction_lower_bounds'])[None, :])),
            prediction_probability_sum_error=float(np.max(np.abs(prior_q.sum(axis=1)-1.))),
            probability_sum_error=float(np.max(np.abs(q.sum(axis=1)-1.))),
            finite_log_weight_underflow_count=int(np.count_nonzero(np.isfinite(logq) & (q == 0.))),
            prior_finite_log_weight_underflow_count=int(np.count_nonzero(np.isfinite(logprior) & (prior_q == 0.))),
            log_transition_underflow_count=int(np.count_nonzero(np.isfinite(self.logtransition) & (self.transition == 0.))),
            exact_unreachable_node_count=int(np.count_nonzero(~np.isfinite(logq))),
            minimum_variance=float(np.min(variance)), negative_variance_roundoff_count=0,
            Gram_symmetry_error=symmetry, posterior_cross_edge_covariance_omitted=True,
            independent_mean_field_edges=True, current_covariance_scope='diagonal_across_edges',
            exact_continuous_posterior_claimed=False, full_history_exact_posterior_claimed=False,
            amplitude_process_changed=True, original_gaussian_transition_equivalence_claimed=False,
            model_grid_min_positive=float(self.values[1]), physical_positive_gap_assumed=False,
            prediction_complexity='O(m*K^2)' if self.mode == 'dense' else 'O(m*K)',
            temporal_complexity='O(m*K^2)' if self.mode == 'dense' else 'O(m*K)',
            branch_order=list(BRANCH_NAMES), initial_temporal_NA=True,
            prediction_seconds=float(prediction_seconds), cavi_seconds=float(cavi_seconds),
            readout_seconds=float(readout_seconds), temporal_seconds=float(temporal_seconds),
            temporal_calls=int(not initial))
        result.update(temporal)
        self.q, self.logq = q.copy(), logq.copy()
        self.step += 1
        result['actor_seconds'] = float(time.perf_counter()-started)
        self.last = copy.deepcopy(result)
        return result

    def snapshot(self):
        return dict(state_version=STATE_VERSION, m=self.m, mode=self.mode, step=self.step,
            model_identity=self.model_identity, model=self.model_receipt(),
            q=self.q.copy(), logq=self.logq.copy(), last=copy.deepcopy(self.last))

    def restore(self, state):
        np, unused = _numeric()
        state = copy.deepcopy(state)
        if (state.get('state_version') != STATE_VERSION or state.get('m') != self.m
                or state.get('mode') != self.mode or state.get('model_identity') != self.model_identity
                or _model_identity(state['model']) != self.model_identity):
            raise ValueError('Checkpoint model, engine, version, or dimensions differ')
        step = state['step']
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError('Checkpoint step must be a nonnegative integer')
        q, logs = np.asarray(state['q']), np.asarray(state['logq'])
        if (q.shape != (self.m, self.k) or logs.shape != q.shape
                or not np.all(np.isfinite(q)) or np.any(q < 0.)
                or np.any(np.isnan(logs)) or np.any(logs == np.inf)
                or np.max(np.abs(q.sum(axis=1)-1.)) > 1e-12
                or np.max(np.abs(np.exp(logs)-q)) > 1e-12):
            raise ValueError('Checkpoint represented probability/log state differs')
        last = state['last']
        if (step == 0) != (last is None):
            raise ValueError('Checkpoint last-output chronology differs')
        if last is not None:
            if (last['step'] != int(step)-1 or last['mode'] != self.mode
                    or not np.array_equal(last['q'], q) or not np.array_equal(last['logq'], logs)):
                raise ValueError('Checkpoint current state and last output differ')
            rebuilt_logs, rebuilt_q = _normalize(np.asarray(last['logprior'])+np.asarray(last['site']))
            if (np.max(np.abs(rebuilt_q-q)) > 1e-12
                    or not np.array_equal(np.isfinite(rebuilt_logs), np.isfinite(logs))):
                raise ValueError('Checkpoint last-used sites and posterior differ')
            finite = np.isfinite(logs)
            if np.any(finite) and np.max(np.abs(rebuilt_logs[finite]-logs[finite])) > 1e-10:
                raise ValueError('Checkpoint last-used sites and log posterior differ')
        self.q, self.logq, self.step, self.last = q.copy(), logs.copy(), int(step), last
        return self


class DenseActor(Actor):
    def __init__(self, m, model):
        super().__init__(m, model, mode='dense')


class FastActor(Actor):
    def __init__(self, m, model):
        super().__init__(m, model, mode='fast')


def setup():
    np, unused = _numeric()
    started = time.perf_counter()
    old_receipt = _load_old().setup()
    kernels = _compiled()
    oldlogs = np.asarray([[math.log(.5), math.log(.3), math.log(.2)], [0., -np.inf, -np.inf]])
    logr = np.log(np.asarray([.6, .4]))
    values = np.asarray([0., .5, 1.])
    logalpha = math.log1p(-DEATH_PROBABILITY)+math.log1p(-RESET_PROBABILITY)
    logbeta = math.log1p(-DEATH_PROBABILITY)+math.log(RESET_PROBABILITY)
    prior_logs, prior_q = kernels['predict'](oldlogs, logr, logalpha, logbeta)
    temporal = kernels['temporal'](oldlogs, prior_logs, prior_logs, prior_q,
        values, logr, logalpha, logbeta)
    if (not np.all(np.isfinite(prior_logs)) or not np.all(np.isfinite(temporal[1]))
            or temporal[-3] > 1e-12 or temporal[-1] > 1e-12):
        raise FloatingPointError('Artificial compiled persistent/reset warmup failed')
    return dict(original_finite_cavi=old_receipt, artificial_only=True,
        warmup_m=2, warmup_K=3, exact_zero_old_support_included=True,
        RNG_used=False, cache=False, fastmath=False, parallel=False,
        fast_backend='lazy_numba', new_helpers_compiled=True,
        wall_seconds=float(time.perf_counter()-started), signatures=signature_inventory())


def signature_inventory():
    predict_signatures = [] if _FAST_KERNELS is None else [str(s) for s in _FAST_KERNELS['predict'].signatures]
    temporal_signatures = [] if _FAST_KERNELS is None else [str(s) for s in _FAST_KERNELS['temporal'].signatures]
    return dict(loaded=_OLD is not None,
        original_finite_cavi={} if _OLD is None else _OLD.signature_inventory(),
        new_helpers_compiled=bool(predict_signatures and temporal_signatures),
        fast_predict=predict_signatures, fast_temporal=temporal_signatures,
        fast_normalize=[] if _FAST_KERNELS is None else [str(s) for s in _FAST_KERNELS['normalize'].signatures],
        fast_backend='lazy_numba', numpy_reference_helpers_available=True)


def binding_receipt():
    old = _load_old()
    return dict(source=str(Path(__file__).resolve()), original_cavi_source=str(OLD_SOURCE.resolve()),
        original_cavi_sha256=_sha(OLD_SOURCE), expected_original_cavi_sha256=OLD_SHA256,
        original_cavi_function=old._cavi.py_func.__name__, max_sweeps=MAX_SWEEPS,
        convergence_tolerance=TOLERANCE, process_model_changed=True,
        dense_and_fast_same_new_T=True, floor_or_clip_used=False,
        fast_backend='lazy_numba', fastmath=False, parallel=False, cache=False,
        model_storage_includes_full_T=True,
        O_mK_scope='prediction_and_temporal_only_not_whole_actor_storage')

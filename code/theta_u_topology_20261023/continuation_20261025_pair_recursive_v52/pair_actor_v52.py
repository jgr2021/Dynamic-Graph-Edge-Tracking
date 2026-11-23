"""Fixed singleton/pair recursive discrete posterior; admitted import only."""
from __future__ import annotations

import math
import time
import numpy as np
from numba import njit
from scipy.special import logsumexp

MAX_SWEEPS = 30
TOLERANCE = 1e-8
PRIMARY_ACTION = 'factorized_support_MAP_conditional_mean'
SECONDARY_ACTION = 'posterior_mean'
TEMPORAL_KEYS = ('old_smoothed_mean', 'old_smoothed_variance', 'old_smoothed_cov',
    'cross_diag', 'cross_cov', 'delta_mean', 'delta_variance', 'delta_cov',
    'temporal_current_marginal_error', 'temporal_conditional_normalization_error',
    'temporal_delta_formula_error', 'temporal_covariance_determinant_minimum')


def _statistics(G, h, m):
    G = np.array(G, dtype=float, order='C', copy=True)
    h = np.array(h, dtype=float, order='C', copy=True)
    if G.shape != (m, m) or h.shape != (m,) or not np.all(np.isfinite(G)) or not np.all(np.isfinite(h)):
        raise ValueError('Finite full candidate-edge G/h required')
    error = float(np.max(np.abs(G-G.T)))
    if error > 1e-10*max(1., float(np.max(np.abs(G)))):
        raise ValueError('Symmetric observation Gram required')
    G = np.ascontiguousarray(.5*(G+G.T))
    if np.any(np.diag(G) < -1e-12):
        raise ValueError('Negative observation Gram diagonal')
    return G, h, error


def _groups(groups, m):
    raw = np.asarray(groups)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('Integer groups[:,2] required; singleton second member=-1')
    answer = np.array(raw, dtype=np.int64, order='C', copy=True)
    covered = []
    for a, b in answer:
        if not 0 <= a < m or b != -1 and not a < b < m:
            raise ValueError('Groups require ascending legal members')
        covered.append(int(a))
        if b >= 0:
            covered.append(int(b))
    if sorted(covered) != list(range(m)) or np.any(np.diff(answer[:, 0]) <= 0):
        raise ValueError('Disjoint complete groups ordered by minimum edge required')
    return answer


def group_offsets(groups, K):
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError('Integer grid size >=2 required')
    groups = np.asarray(groups)
    return np.ascontiguousarray(np.r_[0, np.cumsum(np.where(groups[:, 1] < 0, K, K*K))], dtype=np.int64)


def select_groups(edges, G, paired=True):
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('Integer physical candidate-edge endpoints required')
    m = len(edges)
    if m < 1 or np.any(edges < 0) or np.any(edges[:, 0] >= edges[:, 1]) or len(set(map(tuple, edges))) != m:
        raise ValueError('Distinct ordered physical edges required')
    G, unused, unused_error = _statistics(G, np.zeros(m), m)
    candidates = []
    if paired:
        for e in range(m):
            if G[e, e] <= 0.:
                continue
            for f in range(e+1, m):
                if G[f, f] <= 0. or not any(x == y for x in edges[e] for y in edges[f]):
                    continue
                denominator = math.sqrt(float(G[e, e]*G[f, f]))
                if not math.isfinite(denominator) or denominator <= 0.:
                    raise FloatingPointError('Pair score denominator is not finite positive')
                candidates.append((-abs(float(G[e, f]))/denominator, e, f))
    candidates.sort()
    used, selected = set(), []
    for score, e, f in candidates:
        if e not in used and f not in used:
            selected.append((e, f)); used.update((e, f))
    selected.extend((e, -1) for e in range(m) if e not in used)
    selected.sort(key=lambda group: group[0])
    return _groups(np.asarray(selected, dtype=np.int64), m)


@njit(cache=False, fastmath=False)
def _normalize(logweights):
    maximum = -np.inf
    for j in range(logweights.size):
        maximum = max(maximum, logweights[j])
    if maximum == -np.inf:
        raise FloatingPointError('No finite reachable block state')
    total = 0.
    for j in range(logweights.size):
        total += math.exp(logweights[j]-maximum)
    logarithm = math.log(total)
    logs, probabilities = np.empty_like(logweights), np.empty_like(logweights)
    correction = 0.
    for j in range(logweights.size):
        logs[j] = (logweights[j]-maximum)-logarithm
        probabilities[j] = math.exp(logs[j]); correction += probabilities[j]
    for j in range(logweights.size):
        probabilities[j] /= correction; logs[j] -= math.log(correction)
    return logs, probabilities


@njit(cache=False, fastmath=False)
def _predict_single(logold, logT):
    k = len(logold); result = np.empty(k)
    for j in range(k):
        maximum = -np.inf
        for i in range(k):
            maximum = max(maximum, logold[i]+logT[i, j])
        if maximum == -np.inf:
            result[j] = -np.inf
        else:
            total = 0.
            for i in range(k):
                total += math.exp(logold[i]+logT[i, j]-maximum)
            result[j] = maximum+math.log(total)
    return _normalize(result)[0]


@njit(cache=False, fastmath=False)
def _predict_pair(logold, logT):
    k = logT.shape[0]; logM = np.empty((k, k)); raw = np.empty(k*k)
    for i in range(k):
        for l in range(k):
            maximum = -np.inf
            for j in range(k):
                maximum = max(maximum, logold[i, j]+logT[j, l])
            if maximum == -np.inf:
                logM[i, l] = -np.inf
            else:
                total = 0.
                for j in range(k):
                    total += math.exp(logold[i, j]+logT[j, l]-maximum)
                logM[i, l] = maximum+math.log(total)
    for current_a in range(k):
        for current_b in range(k):
            maximum = -np.inf
            for i in range(k):
                maximum = max(maximum, logT[i, current_a]+logM[i, current_b])
            if maximum == -np.inf:
                raw[current_a*k+current_b] = -np.inf
            else:
                total = 0.
                for i in range(k):
                    total += math.exp(logT[i, current_a]+logM[i, current_b]-maximum)
                raw[current_a*k+current_b] = maximum+math.log(total)
    return _normalize(raw)[0].reshape((k, k)), logM


def predict_pair(logQold, logT):
    logQold = np.array(logQold, dtype=float, order='C', copy=True)
    logT = np.array(logT, dtype=float, order='C', copy=True)
    k = len(logT)
    if logT.shape != (k, k) or logQold.shape != (k, k):
        raise ValueError('K-by-K pair posterior and transition required')
    for value in (logQold, logT):
        if np.any(np.isnan(value)) or np.any(value == np.inf):
            raise ValueError('Only finite or -inf log weights allowed')
    return _predict_pair(logQold, logT)


@njit(cache=False, fastmath=False)
def _prob_marginals(q, groups, offsets, k, m):
    result = np.zeros((m, k))
    for g in range(len(groups)):
        a, b = groups[g, 0], groups[g, 1]; off = offsets[g]
        if b < 0:
            for i in range(k):
                result[a, i] = q[off+i]
        else:
            for i in range(k):
                for j in range(k):
                    mass = q[off+i*k+j]
                    result[a, i] += mass; result[b, j] += mass
    return result


@njit(cache=False, fastmath=False)
def _marginals(q, logq, groups, offsets, k, m):
    marginal = np.zeros((m, k)); logs = np.full((m, k), -np.inf)
    for g in range(len(groups)):
        a, b = groups[g, 0], groups[g, 1]; off = offsets[g]
        if b < 0:
            for i in range(k):
                marginal[a, i] = q[off+i]; logs[a, i] = logq[off+i]
        else:
            for i in range(k):
                maximum_a = -np.inf; maximum_b = -np.inf
                for j in range(k):
                    marginal[a, i] += q[off+i*k+j]
                    marginal[b, i] += q[off+j*k+i]
                    maximum_a = max(maximum_a, logq[off+i*k+j])
                    maximum_b = max(maximum_b, logq[off+j*k+i])
                if maximum_a != -np.inf:
                    total = 0.
                    for j in range(k):
                        total += math.exp(logq[off+i*k+j]-maximum_a)
                    logs[a, i] = maximum_a+math.log(total)
                if maximum_b != -np.inf:
                    total = 0.
                    for j in range(k):
                        total += math.exp(logq[off+j*k+i]-maximum_b)
                    logs[b, i] = maximum_b+math.log(total)
    return marginal, logs


@njit(cache=False, fastmath=False)
def _current_moments(q, groups, offsets, values, m):
    k = len(values); means = np.zeros(m); covariance = np.zeros((m, m))
    for g in range(len(groups)):
        a, b = groups[g, 0], groups[g, 1]; off = offsets[g]
        if b < 0:
            for i in range(k):
                means[a] += q[off+i]*values[i]
            for i in range(k):
                covariance[a, a] += q[off+i]*(values[i]-means[a])**2
        else:
            for i in range(k):
                for j in range(k):
                    mass = q[off+i*k+j]
                    means[a] += mass*values[i]; means[b] += mass*values[j]
            for i in range(k):
                for j in range(k):
                    mass = q[off+i*k+j]; da, db = values[i]-means[a], values[j]-means[b]
                    covariance[a, a] += mass*da*da; covariance[b, b] += mass*db*db
                    covariance[a, b] += mass*da*db
            covariance[b, a] = covariance[a, b]
    return means, covariance


@njit(cache=False, fastmath=False)
def _score(q, logq, logprior, groups, offsets, values, G, h, means):
    m, k = len(means), len(values); score = 0.
    for e in range(m):
        score += h[e]*means[e]
        for f in range(m):
            score -= .5*means[e]*G[e, f]*means[f]
    for g in range(len(groups)):
        a, b = groups[g, 0], groups[g, 1]; off = offsets[g]
        if b < 0:
            for i in range(k):
                mass = q[off+i]
                if mass > 0.:
                    score -= mass*(logq[off+i]-logprior[off+i])
                score -= .5*G[a, a]*mass*(values[i]-means[a])**2
        else:
            for i in range(k):
                for j in range(k):
                    index = off+i*k+j; mass = q[index]
                    if mass > 0.:
                        score -= mass*(logq[index]-logprior[index])
                    da, db = values[i]-means[a], values[j]-means[b]
                    score -= .5*G[a, a]*mass*da*da+.5*G[b, b]*mass*db*db+G[a, b]*mass*da*db
    return score


@njit(cache=False, fastmath=False)
def _cavi(logprior, prior, groups, offsets, values, G, h):
    m, k = len(h), len(values)
    logq, q = logprior.copy(), prior.copy()
    means = _current_moments(q, groups, offsets, values, m)[0]
    residual = h.copy()
    for e in range(m):
        for f in range(m):
            residual[e] -= G[e, f]*means[f]
    sites = np.empty_like(q); cavity = np.empty(m)
    history = np.empty(MAX_SWEEPS+1)
    history[0] = _score(q, logq, logprior, groups, offsets, values, G, h, means)
    converged = bad_elbo = False
    max_mean_delta = max_block_delta = max_marginal_delta = 0.
    start_mean, start_q = means.copy(), q.copy()
    start_marginal = _prob_marginals(q, groups, offsets, k, m)
    for sweep in range(MAX_SWEEPS):
        start_mean, start_q = means.copy(), q.copy()
        start_marginal = _prob_marginals(q, groups, offsets, k, m)
        for g in range(len(groups)):
            a, b = groups[g, 0], groups[g, 1]; off, end = offsets[g], offsets[g+1]
            cavity[a] = residual[a]+G[a, a]*means[a]
            if b >= 0:
                cavity[a] += G[a, b]*means[b]
                cavity[b] = residual[b]+G[b, b]*means[b]+G[b, a]*means[a]
            for i in range(k):
                if b < 0:
                    sites[off+i] = -.5*G[a, a]*values[i]**2+cavity[a]*values[i]
                    logq[off+i] = logprior[off+i]+sites[off+i]
                else:
                    for j in range(k):
                        index = off+i*k+j
                        sites[index] = (-.5*G[a, a]*values[i]**2-.5*G[b, b]*values[j]**2
                            -G[a, b]*values[i]*values[j]+cavity[a]*values[i]+cavity[b]*values[j])
                        logq[index] = logprior[index]+sites[index]
            normalized, probability = _normalize(logq[off:end])
            new_a, new_b = 0., 0.
            for index in range(off, end):
                logq[index] = normalized[index-off]; q[index] = probability[index-off]
            for i in range(k):
                if b < 0:
                    new_a += q[off+i]*values[i]
                else:
                    for j in range(k):
                        mass = q[off+i*k+j]; new_a += mass*values[i]; new_b += mass*values[j]
            change_a = new_a-means[a]; means[a] = new_a
            change_b = 0.
            if b >= 0:
                change_b = new_b-means[b]; means[b] = new_b
            for f in range(m):
                residual[f] -= G[f, a]*change_a
                if b >= 0:
                    residual[f] -= G[f, b]*change_b
        score = _score(q, logq, logprior, groups, offsets, values, G, h, means)
        history[sweep+1] = score
        if score < history[sweep]-1e-9*max(1., abs(score), abs(history[sweep])):
            bad_elbo = True
        marginal = _prob_marginals(q, groups, offsets, k, m)
        max_mean_delta = max_block_delta = max_marginal_delta = 0.
        for e in range(m):
            max_mean_delta = max(max_mean_delta, abs(means[e]-start_mean[e]))
            for j in range(k):
                max_marginal_delta = max(max_marginal_delta, abs(marginal[e, j]-start_marginal[e, j]))
        for index in range(len(q)):
            max_block_delta = max(max_block_delta, abs(q[index]-start_q[index]))
        if max_mean_delta <= TOLERANCE and max(max_block_delta, max_marginal_delta) <= TOLERANCE:
            converged = True
            break
    return (logq, q, sites, cavity, start_mean, start_q, start_marginal,
        history[:sweep+2].copy(), sweep+1, converged, max_mean_delta,
        max_block_delta, max_marginal_delta, bad_elbo)


@njit(cache=False, fastmath=False)
def _finish_temporal(q, values, Bmean, Bcov):
    # q is K (singleton) or flattened K*K (pair); dimensions are at most two.
    d, k = Bmean.shape[1], len(values)
    current = np.zeros(d); old = np.zeros(d)
    for index in range(len(q)):
        x0 = values[index] if d == 1 else values[index//k]
        current[0] += q[index]*x0
        if d == 2:
            current[1] += q[index]*values[index % k]
        for a in range(d):
            old[a] += q[index]*Bmean[index, a]
    Pcur, Pold, cross, delta_cov = (np.zeros((d, d)), np.zeros((d, d)),
                                   np.zeros((d, d)), np.zeros((d, d)))
    delta = current-old
    for index in range(len(q)):
        x = np.empty(d)
        x[0] = values[index] if d == 1 else values[index//k]
        if d == 2:
            x[1] = values[index % k]
        for a in range(d):
            for b in range(d):
                Pcur[a, b] += q[index]*(x[a]-current[a])*(x[b]-current[b])
                Pold[a, b] += q[index]*(Bcov[index, a, b]
                    +(Bmean[index, a]-old[a])*(Bmean[index, b]-old[b]))
                cross[a, b] += q[index]*(x[a]-current[a])*(Bmean[index, b]-old[b])
                delta_cov[a, b] += q[index]*(Bcov[index, a, b]
                    +(x[a]-Bmean[index, a]-delta[a])*(x[b]-Bmean[index, b]-delta[b]))
    identity_error, determinant = 0., np.inf
    for a in range(d):
        determinant = min(determinant, Pcur[a, a]*Pold[a, a]-cross[a, a]**2)
        for b in range(d):
            identity_error = max(identity_error,
                abs(delta_cov[a, b]-(Pcur[a, b]+Pold[a, b]-cross[a, b]-cross[b, a])))
    return current, Pcur, old, Pold, cross, delta, delta_cov, identity_error, determinant


@njit(cache=False, fastmath=False)
def _single_temporal(logold, logT, logprior, q, values):
    k = len(values); Bmean = np.zeros((k, 1)); Bcov = np.zeros((k, 1, 1))
    normalizer_error = current_error = 0.
    weights = np.empty(k)
    for j in range(k):
        maximum = -np.inf
        for i in range(k):
            maximum = max(maximum, logold[i]+logT[i, j])
        if maximum == -np.inf:
            if q[j] > 0.:
                raise FloatingPointError('Current mass at unreachable singleton node')
            continue
        total = 0.
        for i in range(k):
            weights[i] = math.exp(logold[i]+logT[i, j]-maximum); total += weights[i]
        raw_logsum = maximum+math.log(total)
        if np.isfinite(logprior[j]):
            normalizer_error = max(normalizer_error, abs(math.expm1(raw_logsum-logprior[j])))
        elif q[j] > 0.:
            raise FloatingPointError('Current singleton mass outside predicted support')
        norm, running, mean, second = 0., 0., 0., 0.
        for i in range(k):
            weight = weights[i]/total; norm += weight
            if weight > 0.:
                new_total = running+weight; difference = values[i]-mean
                new_mean = mean+(weight/new_total)*difference
                second += weight*difference*(values[i]-new_mean)
                running, mean = new_total, new_mean
        Bmean[j, 0] = mean; Bcov[j, 0, 0] = second/running
        current_error = max(current_error, abs(q[j]*norm-q[j]))
    answer = _finish_temporal(q, values, Bmean, Bcov)
    return answer, current_error, normalizer_error


@njit(cache=False, fastmath=False)
def _pair_temporal(logold, logT, logprior, q, values, logM):
    k = len(values); Jmean = np.zeros((k, k)); Jvar = np.zeros((k, k))
    # j | i,l. One weighted centered pass per (i,l), never a K^4 tensor.
    for i in range(k):
        for l in range(k):
            if logM[i, l] == -np.inf:
                continue
            total, mean, second = 0., 0., 0.
            for j in range(k):
                weight = math.exp(logold[i, j]+logT[j, l]-logM[i, l])
                if weight > 0.:
                    new_total = total+weight; difference = values[j]-mean
                    new_mean = mean+(weight/new_total)*difference
                    second += weight*difference*(values[j]-new_mean)
                    total, mean = new_total, new_mean
            if total <= 0.:
                raise FloatingPointError('Reachable conditional old-node row lost all mass')
            Jmean[i, l] = mean; Jvar[i, l] = second/total
    Bmean = np.zeros((k*k, 2)); Bcov = np.zeros((k*k, 2, 2))
    current_error = normalizer_error = 0.; weights = np.empty(k)
    # i | k,l, then the within-old conditional covariance mixture.
    for current_a in range(k):
        for current_b in range(k):
            index = current_a*k+current_b; maximum = -np.inf
            for i in range(k):
                maximum = max(maximum, logT[i, current_a]+logM[i, current_b])
            if maximum == -np.inf:
                if q[index] > 0.:
                    raise FloatingPointError('Current mass at unreachable pair node')
                continue
            total = 0.
            for i in range(k):
                weights[i] = math.exp(logT[i, current_a]+logM[i, current_b]-maximum)
                total += weights[i]
            raw_logsum = maximum+math.log(total)
            if np.isfinite(logprior[index]):
                normalizer_error = max(normalizer_error, abs(math.expm1(raw_logsum-logprior[index])))
            elif q[index] > 0.:
                raise FloatingPointError('Current pair mass outside predicted support')
            running, mean1, mean2 = 0., 0., 0.
            second11, second12, second22, norm = 0., 0., 0., 0.
            for i in range(k):
                weight = weights[i]/total; norm += weight
                if weight > 0.:
                    new_total = running+weight
                    d1, d2 = values[i]-mean1, Jmean[i, current_b]-mean2
                    new1 = mean1+(weight/new_total)*d1
                    new2 = mean2+(weight/new_total)*d2
                    second11 += weight*d1*(values[i]-new1)
                    second12 += weight*d1*(Jmean[i, current_b]-new2)
                    second22 += weight*(d2*(Jmean[i, current_b]-new2)+Jvar[i, current_b])
                    running, mean1, mean2 = new_total, new1, new2
            Bmean[index, 0], Bmean[index, 1] = mean1, mean2
            Bcov[index, 0, 0], Bcov[index, 1, 1] = second11/running, second22/running
            Bcov[index, 0, 1] = second12/running; Bcov[index, 1, 0] = Bcov[index, 0, 1]
            current_error = max(current_error, abs(q[index]*norm-q[index]))
    answer = _finish_temporal(q, values, Bmean, Bcov)
    return answer, current_error, normalizer_error


def _temporal_dictionary(answer, current_error, normalizer_error):
    current, Pcur, old, Pold, cross, delta, Pdelta, identity, determinant = answer
    return dict(current_mean=current, current_cov=Pcur, old_smoothed_mean=old,
        old_smoothed_cov=Pold, cross_cov=cross, delta_mean=delta, delta_cov=Pdelta,
        temporal_current_marginal_error=float(current_error),
        temporal_conditional_normalization_error=float(normalizer_error),
        temporal_delta_formula_error=float(identity),
        temporal_covariance_determinant_minimum=float(determinant))


def pair_temporal(logold, logT, logprior, qnew, values, logM=None):
    values = np.array(values, dtype=float, order='C', copy=True); k = len(values)
    logold, logT, logprior, qnew = [np.array(x, dtype=float, order='C', copy=True)
                                    for x in (logold, logT, logprior, qnew)]
    if any(x.shape != (k, k) for x in (logold, logT, logprior, qnew)):
        raise ValueError('K-by-K temporal pair inputs required')
    for logs in (logold, logT, logprior):
        if np.any(np.isnan(logs)) or np.any(logs == np.inf):
            raise ValueError('Temporal logarithms must be finite or -inf')
    if not np.all(np.isfinite(qnew)) or np.any(qnew < 0.) or abs(float(qnew.sum()-1.)) > 1e-12:
        raise ValueError('Normalized current pair mass required')
    if logM is None:
        unused, logM = _predict_pair(logold, logT)
    else:
        logM = np.array(logM, dtype=float, order='C', copy=True)
        if logM.shape != (k, k) or np.any(np.isnan(logM)) or np.any(logM == np.inf):
            raise ValueError('Valid K-by-K cached logM required')
    answer, current_error, normalizer = _pair_temporal(logold, logT,
        logprior.reshape(k*k), qnew.reshape(k*k), values, logM)
    return _temporal_dictionary(answer, current_error, normalizer)


def _values(values, k=None):
    if values is None:
        if k is None or k < 2:
            raise ValueError('Grid size >=2 required')
        values = np.arange(k, dtype=float)/(k-1)
    values = np.array(values, dtype=float, order='C', copy=True)
    if (values.ndim != 1 or len(values) < 2 or k is not None and len(values) != k
            or not np.all(np.isfinite(values)) or values[0] != 0.
            or values[-1] > 1. or np.any(np.diff(values) <= 0.)):
        raise ValueError('Actual finite increasing zero-based weight grid required')
    return values


def _edge_prior(logprior, m, k):
    result = np.array(logprior, dtype=float, order='C', copy=True)
    if (result.shape != (m, k) or np.any(np.isnan(result)) or np.any(result == np.inf)
            or not np.all(np.any(np.isfinite(result), axis=1))):
        raise ValueError('Finite or unreachable full edge prior required')
    return result


def _flat_from_edges(logprior, groups, offsets, k):
    logs = np.empty(int(offsets[-1])); probability = np.empty_like(logs)
    for g, (a, b) in enumerate(groups):
        raw = logprior[a].copy() if b < 0 else (logprior[a, :, None]+logprior[b, None, :]).reshape(k*k)
        left, right = offsets[g], offsets[g+1]
        logs[left:right], probability[left:right] = _normalize(np.ascontiguousarray(raw))
    return logs, probability


def _predict_flat(logold, groups, offsets, logT, k):
    logs = np.empty_like(logold); probability = np.empty_like(logold); cache = []
    for g, (a, b) in enumerate(groups):
        left, right = offsets[g], offsets[g+1]
        if b < 0:
            logs[left:right] = _predict_single(logold[left:right], logT); cache.append(None)
        else:
            predicted, intermediate = _predict_pair(logold[left:right].reshape((k, k)), logT)
            logs[left:right] = predicted.reshape(k*k); cache.append(intermediate)
        probability[left:right] = np.exp(logs[left:right])
    return logs, probability, cache


def _variance_audit(matrix, label):
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError('Nonfinite '+label)
    diagonal = np.diag(matrix)
    tolerance = 1e-12*max(1., float(np.max(np.abs(matrix))))
    if np.min(diagonal) < -tolerance:
        raise FloatingPointError('Negative variance beyond rounding tolerance: '+label)
    # Preserve, count and report small negative values; do not clip them.
    return int(np.count_nonzero(diagonal < 0.))


def _current_result(logprior, prior, groups, offsets, values, G, h):
    clock = time.perf_counter()
    (logq, q, sites, cavity, start_mean, start_q, start_marginal, history,
     sweeps, converged, mean_delta, block_delta, marginal_delta, bad_elbo) = _cavi(
        logprior, prior, groups, offsets, values, G, h)
    cavi_seconds = time.perf_counter()-clock
    if bad_elbo:
        raise FloatingPointError('Block coordinate ELBO decreased beyond frozen tolerance')
    clock = time.perf_counter()
    m, k = len(h), len(values)
    marginal, marginal_logs = _marginals(q, logq, groups, offsets, k, m)
    prior_marginal, prior_logs = _marginals(prior, logprior, groups, offsets, k, m)
    means, covariance = _current_moments(q, groups, offsets, values, m)
    negative_count = _variance_audit(covariance, 'current_cov')
    if (not np.all(np.isfinite(q)) or not np.all(np.isfinite(means))
            or not np.all(np.isfinite(history)) or not np.all(np.isfinite(sites))
            or np.any(np.isnan(logq)) or np.any(logq == np.inf)
            or np.any(q < 0.) or np.max(np.abs(marginal.sum(axis=1)-1.)) > 1e-12):
        raise FloatingPointError('Invalid complete block posterior')
    moment_seconds = time.perf_counter()-clock
    clock = time.perf_counter()
    support = np.empty(m); positive = []
    for e in range(m):
        logmass = float(logsumexp(marginal_logs[e, 1:]))
        if np.isfinite(logmass):
            mass = np.exp(marginal_logs[e, 1:]-logmass); mass /= float(mass.sum())
            value = float(mass @ values[1:]); positive.append(value)
        else:
            value = None; positive.append(None)
        if marginal[e, 0] >= .5:
            support[e] = 0.
        elif value is None:
            raise FloatingPointError('Positive support selected without reachable positive state')
        else:
            support[e] = value
    if np.any((means < 0.) | (means > 1.)) or not np.all(np.isfinite(support)) or np.any((support < 0.) | (support > 1.)):
        raise FloatingPointError('Illegal full candidate-edge readout')
    readout_seconds = time.perf_counter()-clock
    result = dict(q=marginal, logq=marginal_logs, logprior=prior_logs, prior_q=prior_marginal,
        flat_logq=logq, flat_q=q, flat_logprior=logprior, flat_prior_q=prior,
        flat_logold=None, groups=groups.copy(), offsets=offsets.copy(),
        site_flat=sites, cavity_scalar=cavity,
        last_sweep_start_mean=start_mean, last_sweep_start_q_flat=start_q,
        last_sweep_start_q=start_marginal, G=G, h=h,
        current_mean=means, current_cov=covariance,
        mean_action=means.copy(), support_action=support, variance=np.diag(covariance).copy(),
        old_q=None, logold=None, old_filtered_mean=None,
        positive_conditional_mean=positive, sweeps=int(sweeps),
        converged=bool(converged), qualification_passed=bool(converged),
        max_mean_delta=float(mean_delta), max_q_delta=float(max(block_delta, marginal_delta)),
        max_block_q_delta=float(block_delta), max_marginal_q_delta=float(marginal_delta),
        convergence_rule='max_mean_and_max_block_and_marginal_probability_delta_le_1e-8',
        max_sweeps=MAX_SWEEPS, convergence_tolerance=TOLERANCE,
        elbo_history=history.tolist(), bad_elbo=bool(bad_elbo),
        initial=True, step=0, primary_action=PRIMARY_ACTION, secondary_action=SECONDARY_ACTION,
        finite_log_weight_underflow_count=int(np.count_nonzero(np.isfinite(logq) & (q == 0.))),
        exact_unreachable_node_count=int(np.count_nonzero(~np.isfinite(logq))),
        probability_sum_error=float(np.max(np.abs(marginal.sum(axis=1)-1.))),
        minimum_variance=float(np.min(np.diag(covariance))),
        negative_variance_roundoff_count=negative_count,
        independent_mean_field_blocks=True, independent_mean_field_edges=bool(np.all(groups[:, 1] < 0)),
        posterior_cross_edge_covariance_omitted=True, current_covariance_scope='block_diagonal',
        exact_continuous_posterior_claimed=False, full_history_exact_posterior_claimed=False,
        model_grid_min_positive=float(values[1]), physical_positive_gap_assumed=False,
        cavi_seconds=float(cavi_seconds), current_moment_seconds=float(moment_seconds),
        readout_seconds=float(readout_seconds), temporal_seconds=0., temporal_calls=0)
    result.update({key: None for key in TEMPORAL_KEYS})
    return result


def _temporal_full(logold, logT, logprior, qnew, groups, offsets, values, cache, current_means, current_cov):
    m, k = len(current_means), len(values)
    old = np.zeros(m); old_cov = np.zeros((m, m)); cross = np.zeros((m, m))
    delta = np.zeros(m); delta_cov = np.zeros((m, m))
    current_error = normalizer_error = identity_error = 0.; determinant = np.inf
    for g, (a, b) in enumerate(groups):
        left, right = offsets[g], offsets[g+1]
        if b < 0:
            answer, ce, ne = _single_temporal(logold[left:right], logT,
                logprior[left:right], qnew[left:right], values)
            members = (int(a),)
        else:
            answer, ce, ne = _pair_temporal(logold[left:right].reshape((k, k)), logT,
                logprior[left:right], qnew[left:right], values, cache[g])
            members = (int(a), int(b))
        cur, Pcur, old_g, Pold, C, delta_g, Pdelta, error, det = answer
        current_error = max(current_error, float(ce)); normalizer_error = max(normalizer_error, float(ne))
        identity_error = max(identity_error, float(error)); determinant = min(determinant, float(det))
        for i, e in enumerate(members):
            old[e] = old_g[i]; delta[e] = delta_g[i]
            if abs(float(cur[i]-current_means[e])) > 1e-10:
                raise FloatingPointError('Temporal/current mean assembly differs')
            for j, f in enumerate(members):
                old_cov[e, f] = Pold[i, j]; cross[e, f] = C[i, j]; delta_cov[e, f] = Pdelta[i, j]
                if abs(float(Pcur[i, j]-current_cov[e, f])) > 1e-10:
                    raise FloatingPointError('Temporal/current covariance assembly differs')
    negative_count = _variance_audit(old_cov, 'old_smoothed_cov')+_variance_audit(delta_cov, 'delta_cov')
    identity_error = max(identity_error, float(np.max(np.abs(delta_cov-(current_cov+old_cov-cross-cross.T)))))
    if not np.all(np.isfinite(cross)) or not np.all(np.isfinite(old)) or not np.all(np.isfinite(delta)):
        raise FloatingPointError('Nonfinite block temporal moments')
    return dict(old_smoothed_mean=old, old_smoothed_cov=old_cov,
        old_smoothed_variance=np.diag(old_cov).copy(), cross_cov=cross, cross_diag=np.diag(cross).copy(),
        delta_mean=delta, delta_cov=delta_cov, delta_variance=np.diag(delta_cov).copy(),
        temporal_current_marginal_error=current_error,
        temporal_conditional_normalization_error=normalizer_error,
        temporal_delta_formula_error=identity_error,
        temporal_covariance_determinant_minimum=determinant,
        temporal_negative_variance_roundoff_count=negative_count)


class Actor:
    def __init__(self, edges, model, paired=True):
        raw_edges = np.asarray(edges)
        if not np.issubdtype(raw_edges.dtype, np.integer):
            raise ValueError('Integer physical candidate-edge endpoints required')
        self.edges = np.array(edges, dtype=np.int64, order='C', copy=True)
        if self.edges.ndim != 2 or self.edges.shape[1] != 2 or len(self.edges) < 1:
            raise ValueError('Physical full candidate-edge array required')
        self.m = len(self.edges); self.paired = bool(paired)
        self._model = {key: value.copy() if isinstance(value, np.ndarray) else value
                       for key, value in model.items()}
        self.values = _values(model['values']); self.k = len(self.values)
        self.logtransition = np.array(model['logtransition'], dtype=float, order='C', copy=True)
        self.loginitial = np.array(model['loginitial'], dtype=float, order='C', copy=True)
        if self.logtransition.shape != (self.k, self.k) or self.loginitial.shape != (self.k,):
            raise ValueError('Actual model logarithmic dimensions differ')
        for logs in (self.logtransition, self.loginitial):
            if np.any(np.isnan(logs)) or np.any(logs == np.inf):
                raise ValueError('Model logs must be finite or -inf')
        if (np.max(np.abs(logsumexp(self.logtransition, axis=1))) > 1e-12
                or abs(float(logsumexp(self.loginitial))) > 1e-12):
            raise ValueError('Normalized fixed Markov model required')
        self.transition = np.exp(self.logtransition)
        self.groups = self.offsets = self.flat_logq = self.flat_q = None
        self.logq = np.tile(self.loginitial, (self.m, 1)); self.q = np.exp(self.logq)
        self.step = 0; self.last = None

    def model_receipt(self):
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self._model.items()}

    def update(self, G, h):
        started = time.perf_counter(); clock = started
        G, h, symmetry = _statistics(G, h, self.m)
        validation_seconds = time.perf_counter()-clock
        clock = time.perf_counter(); initial = self.step == 0
        if initial:
            self.groups = select_groups(self.edges, G, self.paired)
            self.offsets = group_offsets(self.groups, self.k)
        selection_seconds = time.perf_counter()-clock
        clock = time.perf_counter()
        oldlogs = None if initial else self.flat_logq.copy()
        oldq = None if initial else self.flat_q.copy()
        old_marginal = None if initial else self.q.copy()
        old_marginal_logs = None if initial else self.logq.copy()
        if initial:
            logs, prior = _flat_from_edges(np.tile(self.loginitial, (self.m, 1)), self.groups, self.offsets, self.k)
            cache = None
        else:
            logs, prior, cache = _predict_flat(oldlogs, self.groups, self.offsets, self.logtransition, self.k)
        prediction_seconds = time.perf_counter()-clock
        result = _current_result(logs, prior, self.groups, self.offsets, self.values, G, h)
        clock = time.perf_counter()
        if not initial:
            result.update(_temporal_full(oldlogs, self.logtransition, logs, result['flat_q'],
                self.groups, self.offsets, self.values, cache, result['current_mean'], result['current_cov']))
            result.update(old_q=old_marginal, logold=old_marginal_logs, flat_logold=oldlogs,
                          old_filtered_mean=_current_moments(oldq, self.groups, self.offsets, self.values, self.m)[0])
            result['temporal_seconds'] = float(time.perf_counter()-clock)
        else:
            result['temporal_negative_variance_roundoff_count'] = None
            result['temporal_seconds'] = 0.
        result['temporal_calls'] = int(not initial)
        result.update(initial=initial, step=self.step, Gram_symmetry_error=symmetry,
            validation_seconds=float(validation_seconds), selection_seconds=float(selection_seconds),
            prediction_seconds=float(prediction_seconds), group_count=len(self.groups),
            pair_count=int(np.count_nonzero(self.groups[:, 1] >= 0)), flat_state_count=int(self.offsets[-1]),
            grouping_selected_only_from_first_G=True, paired_requested=self.paired,
            log_transition_underflow_count=int(np.count_nonzero(np.isfinite(self.logtransition) & (self.transition == 0.))))
        self.flat_logq, self.flat_q = result['flat_logq'].copy(), result['flat_q'].copy()
        self.logq, self.q = result['logq'].copy(), result['q'].copy()
        self.step += 1
        result['actor_seconds'] = float(time.perf_counter()-started)
        self.last = result
        return result


def fit_current(logprior_edges, G, h, groups=None, values=None):
    """Mechanism control at the actual saved independent edge prediction only."""
    started = time.perf_counter()
    raw = np.asarray(logprior_edges)
    if raw.ndim != 2:
        raise ValueError('Full m-by-K actual predicted edge logs required')
    m, k = raw.shape; values = _values(values, k)
    G, h, symmetry = _statistics(G, h, m)
    groups = _groups(np.column_stack((np.arange(m), np.full(m, -1))), m) if groups is None else _groups(groups, m)
    offsets = group_offsets(groups, k)
    clock = time.perf_counter()
    logprior = _edge_prior(logprior_edges, m, k)
    logs, prior = _flat_from_edges(logprior, groups, offsets, k)
    prediction_seconds = time.perf_counter()-clock
    result = _current_result(logs, prior, groups, offsets, values, G, h)
    result.update(Gram_symmetry_error=symmetry, validation_seconds=0., selection_seconds=0.,
        prediction_seconds=float(prediction_seconds), temporal_negative_variance_roundoff_count=None,
        group_count=len(groups), pair_count=int(np.count_nonzero(groups[:, 1] >= 0)),
        flat_state_count=int(offsets[-1]), current_target_only=True,
        grouping_selected_only_from_first_G=False, grouping_supplied_by_caller=True,
        log_transition_underflow_count=None, actor_seconds=float(time.perf_counter()-started))
    return result


def signature_inventory():
    names = ('_normalize', '_predict_single', '_predict_pair', '_prob_marginals', '_marginals',
             '_current_moments', '_score', '_cavi', '_finish_temporal', '_single_temporal', '_pair_temporal')
    return {name: [str(s) for s in globals()[name].signatures] for name in names}


def setup():
    started = time.perf_counter()
    values = np.asarray([0., .3, .7]); initial = np.asarray([.5, .3, .2])
    transition = np.asarray([[.8, .1, .1], [.1, .85, .05], [.1, .05, .85]])
    model = dict(values=values, loginitial=np.log(initial), logtransition=np.log(transition))
    edges = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    G = np.asarray([[4., .8, .2], [.8, 3., .3], [.2, .3, 2.]])
    h = np.asarray([.2, .9, .5]); histories = {}
    for name, paired in (('single', False), ('pair', True)):
        actor = Actor(edges, model, paired)
        history = []
        for t in range(2):
            result = actor.update(G, h)
            if not result['converged']:
                raise RuntimeError('Artificial block actor warmup did not converge')
            history.append(dict(step=t, sweeps=result['sweeps'], pair_count=result['pair_count']))
        histories[name] = history
    return dict(artificial_only=True, scientific_observations_used=False, RNG_used=False,
        m=3, K=3, steps_per_variant=2, histories=histories,
        signatures=signature_inventory(), wall_seconds=float(time.perf_counter()-started))

"""Owned causal V60 pipelines. Import defines interfaces; it runs no numerics.

Every filter owns its statistics plan, copied model and entire state history.
Only the saved node-observation whitelist enters an update. V53 comparisons are
qualification checks made after the corresponding private updates have run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

import common as c

ROOT, PROJECT = c.ROOT, c.PROJECT
CHAINS = ('old', 'dense', 'fast')
BASELINES = ('gaussian', 'hard_box')
FILTERS = CHAINS + BASELINES
METHODS = ('old_support', 'old_mean', 'dense_support', 'dense_mean',
           'fast_support', 'fast_mean', 'gaussian', 'hard_box')
MODES = ('exact_eigh', 'warm_jacobi')
V53 = PROJECT/'continuation_20261025_author_baselines_v53'
V16 = PROJECT/'continuation_20261024_fixed_star_development_v16'
V21 = PROJECT/'continuation_20261024_incidence_statistics_v21'
V24 = PROJECT/'continuation_20261024_gaussian_center_development_v24'
T = 72
FIXED_REQUIRED = ('q', 'logq', 'prior_q', 'logprior', 'current_mean',
    'current_cov', 'variance', 'mean_action', 'support_action', 'G', 'h',
    'cavity_scalar', 'last_sweep_start_mean', 'last_sweep_start_q')
TEMPORAL_ARRAYS = ('old_smoothed_mean', 'old_smoothed_variance',
    'old_smoothed_cov', 'cross_diag', 'cross_cov', 'delta_mean',
    'delta_variance', 'delta_cov', 'old_q', 'logold', 'old_filtered_mean')
EQUIVALENCE_KEYS = FIXED_REQUIRED + TEMPORAL_ARRAYS + ('old_smoothed_q', 'old_smoothed_logq', 'site')


def load_numeric():
    """Load and pin admitted project modules, after the caller's admission."""
    global np, atomic, kernel_actor, fast, l0, spectral, Gaussian, Author
    global pools, limits, MODULE_ORIGINS, OldActor
    import numpy as np
    from threadpoolctl import threadpool_info as pools, threadpool_limits as limits
    for path in (V53, V24, V21, V16):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    atomic = importlib.import_module('actors_v53')
    if Path(atomic.__file__).resolve() != (V53/'actors_v53.py').resolve():
        raise ImportError('Mixed V53 singleton helper origin')
    atomic.load_modules()
    # This is the exact class instantiated by actors_v53.make_actor. Supplying
    # the admitted model explicitly avoids an uncharged default-model copy.
    OldActor = atomic.OLD.NEW.Actor
    kernel_actor = importlib.import_module('kernel_actor_v60')
    # Force the core's lazy, hash-pinned original finite-CAVI binding before
    # inventorying every actually loaded project module.
    kernel_actor.binding_receipt()
    fast = importlib.import_module('fast_statistics_v21')
    l0 = importlib.import_module('l0_kf')
    spectral = importlib.import_module('spectral')
    gaussian_module = importlib.import_module('center_kf_v24')
    author_module = importlib.import_module('author_actor_v53')
    Gaussian, Author = gaussian_module.GaussianCenterKF, author_module.Actor
    for module, expected in ((kernel_actor, ROOT/'kernel_actor_v60.py'),
            (fast, V21/'fast_statistics_v21.py'), (l0, V16/'l0_kf.py'),
            (spectral, V16/'spectral.py'), (gaussian_module, V24/'center_kf_v24.py'),
            (author_module, V53/'author_actor_v53.py')):
        if Path(module.__file__).resolve() != expected.resolve():
            raise ImportError('Mixed canonical dependency: '+str(expected))
    freeze = c.read(ROOT/'FREEZE.json')
    MODULE_ORIGINS = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(PROJECT):
            continue
        relative = path.relative_to(PROJECT).as_posix()
        expected = (freeze['sources'].get(path.name) if path.parent == ROOT
                    else freeze['inputs'].get(relative))
        if expected is None or c.sha(path) != expected:
            raise ImportError('Loaded project source is not admitted: '+name+' '+relative)
        MODULE_ORIGINS[name] = dict(path=str(path), sha256=expected)
    return MODULE_ORIGINS


def signatures():
    result = dict(atomic=atomic.signature_inventory())
    if hasattr(kernel_actor, 'signature_inventory'):
        result['kernel_actor'] = kernel_actor.signature_inventory()
    for module in (kernel_actor, fast, l0, spectral._old):
        for name, value in vars(module).items():
            if hasattr(value, 'signatures'):
                result[module.__name__+'.'+name] = [str(s) for s in value.signatures]
    return result


def structure(n):
    edges = np.asarray(list(itertools.combinations(range(n), 2)), dtype=np.int64)
    obj = fast.EdgeObservationStructure(n, edges)
    B = np.zeros((n, len(edges)))
    B[edges[:, 0], np.arange(len(edges))] = 1.
    B[edges[:, 1], np.arange(len(edges))] = -1.
    return obj, B


def spectrum_step(pair, mode, L, theta):
    (values, U), meta = pair.trackers[mode].update(L, theta)
    values, U, meta = pair._strict_final(mode, L, theta, values, U, meta)
    pair.cache[mode] = (values, U)
    pair.last_laplacian = L
    return values, U, meta


def source_path(relative):
    path = (PROJECT/relative).resolve()
    if PROJECT.resolve() not in path.parents:
        raise ValueError('Input path escaped project')
    return path


def model_load(protocol):
    """Load only the frozen finite-model NPZ, without deriving a new grid."""
    with np.load(source_path(protocol['model_source']), allow_pickle=False) as raw:
        model = {key: np.array(raw[key], copy=True) for key in raw.files}
    for key in ('values', 'logtransition', 'loginitial'):
        if key not in model:
            raise ValueError('Missing admitted model array: '+key)
    if model['values'].shape != (65,) or model['logtransition'].shape != (65, 65):
        raise ValueError('Original K65 finite model required')
    return model


def setup(protocol=None):
    """Small artificial-only JIT/library warmup, paid separately from histories."""
    if protocol is None:
        protocol = c.read(ROOT/'PROTOCOL_V60.json')
    started = time.perf_counter()
    receipts = dict(atomic=atomic.setup(), kernel=kernel_actor.setup(),
                    l0=l0.setup(), statistics=fast.setup())
    obj, B = structure(4)
    pair = spectral.SpectrumPair(4, B)
    receipts['spectral'] = pair.setup()
    X = np.ascontiguousarray(np.sin(np.arange(48.).reshape(12, 4)*.17))
    response = np.ascontiguousarray(.2*X@obj.laplacian(np.full(obj.m, .3)))
    node = fast.NodeStatistics.from_observations(obj, X, response, .2, .03)
    stats = fast.IncidenceStatisticsPlan(obj).expand(node)
    model = model_load(protocol)
    for name in FILTERS:
        if name == 'old':
            actor = OldActor(obj.edges, copy.deepcopy(model), paired=False)
        elif name in ('dense', 'fast'):
            actor = kernel_actor.Actor(obj.m, kernel_actor.make_model(copy.deepcopy(model)), mode=name)
        elif name == 'gaussian':
            actor = Gaussian(obj.m)
        else:
            actor = Author(obj.m, kind='hard', tau=.1, box=True)
        for unused in range(2):
            if name in CHAINS:
                current = actor.update(stats.G, stats.h)
                point = current['support_action']
            else:
                point, meta = actor.update(stats)
                if not meta['success']:
                    raise RuntimeError('Artificial baseline warmup failed')
            for mode in MODES:
                spectrum_step(pair, mode, obj.laplacian(point), point)
            pair.t += 1
    return dict(receipts=receipts, wall_seconds=time.perf_counter()-started,
        signatures=signatures(), artificial_only=True, scientific_observations_used=False,
        module_origins=MODULE_ORIGINS, old_binding=atomic.binding_receipt())


def npz(path, arrays):
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise TypeError('Object array prohibited: '+key)
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)


def sanitize(case, destination=None):
    with np.load(source_path(case['source_observations']), allow_pickle=False) as raw:
        observed = {key: np.array(raw[key], copy=True)
                    for key in ('states', 'inputs', 'a', 'eta', 'sigma')}
    if (observed['states'].shape != (T, 49, case['n'])
            or observed['inputs'].shape != (T, 48, case['n'])):
        raise ValueError('Complete original 72-block all-node history required')
    if any(not np.all(np.isfinite(value)) for value in observed.values()):
        raise FloatingPointError('Nonfinite observed node arrays')
    if (float(observed['a']) != .95 or float(observed['eta']) != .8/case['n']
            or float(observed['sigma']) != .03):
        raise ValueError('Original likelihood constants differ')
    if destination is not None:
        npz(destination, observed)
    return observed


def statistics_for(obj, plan, observations, t):
    X = np.ascontiguousarray(observations['states'][t, :-1])
    response = np.ascontiguousarray(float(observations['a'])*X
        +observations['inputs'][t]-observations['states'][t, 1:])
    node = fast.NodeStatistics.from_observations(obj, X, response,
        float(observations['eta']), float(observations['sigma']))
    return plan.expand(node), X, response


def dense_loader(obj, B, X, response, eta, sigma):
    def build():
        H = np.concatenate([eta*B*(B.T@x)[None, :] for x in X], axis=0)
        return np.ascontiguousarray(H), np.ascontiguousarray(response.reshape(-1)), sigma
    return build


def scalar_meta(result):
    """Preserve scalar/list diagnostics; arrays have an explicit NPZ schema."""
    answer = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, dict):
            answer[key] = scalar_meta(value)
        else:
            answer[key] = value
    answer['none_fields'] = sorted(key for key, value in result.items() if value is None)
    return answer


def append_fixed(store, name, value):
    if value is None:
        raise ValueError('Fixed-schema array may not be None: '+name)
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise TypeError('Object fixed-schema array: '+name)
    store.setdefault(name, []).append(value.copy())


def fingerprint_metadata(result):
    fields = sorted(key for key,value in result.items() if isinstance(value,np.ndarray))
    return dict(fingerprint_array_fields=fields,
        fingerprint_array_dtypes={key:str(result[key].dtype) for key in fields},
        fingerprint_none_fields=sorted(key for key,value in result.items() if value is None))


def capture_raw_arrays(store, prefix, result, t, exclude=()):
    for key,value in result.items():
        if isinstance(value,np.ndarray) and key not in exclude:
            if value.dtype.hasobject:
                raise TypeError('Object core output: '+key)
            store.setdefault(prefix+'__'+key,[None]*T)[t]=value.copy()


def decision_state(result):
    return dict(step=int(result['step']),
        iterations=int(result.get('iterations',result['sweeps'])),
        cap=bool(not result['converged']), converged=bool(result['converged']),
        initial=bool(result['initial']))


def capture_atomic(store, name, result, t):
    capture_raw_arrays(store,name,result,t)
    store.setdefault(name+'__current_variance',[None]*T)[t]=result['variance'].copy()
    store.setdefault(name+'__adjacent_pair_indices',[None]*T)[t]=(np.asarray([t-1,t],dtype=float) if t else np.full(2,np.nan))
    if name=='old':store.setdefault(name+'__site',[None]*T)[t]=result['site_flat'].reshape(result['q'].shape).copy()


def stack_arrays(store):
    answer={}
    for key,values in store.items():
        if len(values)!=T:raise ValueError('Incomplete time stack: '+key)
        example=next((value for value in values if value is not None),None)
        if example is None:continue
        if any(value is not None and np.asarray(value).shape!=example.shape for value in values):raise ValueError('Changing array shape: '+key)
        answer[key]=np.stack([np.full(example.shape,np.nan) if value is None else value for value in values])
    return answer


def flatten_snapshot(value, arrays, key='state'):
    """Lossless no-pickle nested state representation for snapshot/restore."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object checkpoint array')
        arrays[key] = value.copy()
        return {'array': key}
    if isinstance(value, dict):
        return {'dict': {str(k): flatten_snapshot(v, arrays, key+'__'+str(k))
                         for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {'sequence': [flatten_snapshot(v, arrays, key+'__'+str(i))
                             for i, v in enumerate(value)],
                'tuple': isinstance(value, tuple)}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return {'scalar': value}
    raise TypeError('Unsupported snapshot field: '+type(value).__name__)


def unflatten_snapshot(schema, arrays):
    if 'array' in schema:
        return np.array(arrays[schema['array']], copy=True)
    if 'dict' in schema:
        return {k: unflatten_snapshot(v, arrays) for k, v in schema['dict'].items()}
    if 'sequence' in schema:
        values = [unflatten_snapshot(v, arrays) for v in schema['sequence']]
        return tuple(values) if schema['tuple'] else values
    return schema['scalar']


def hash_arrays(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        value = values[key]
        digest.update(key.encode('utf-8')+b'\0')
        if value is None:
            digest.update(b'NONE')
            continue
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def array_fingerprint(result):
    return hash_arrays({key: value for key, value in result.items()
                        if isinstance(value, np.ndarray) or value is None})


def difference(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        return float('inf')
    if not np.array_equal(actual[~np.isfinite(actual)], expected[~np.isfinite(expected)]):
        return float('inf')
    mask = np.isfinite(actual)
    if actual.dtype.kind == 'b' or expected.dtype.kind == 'b':
        if actual.dtype.kind != 'b' or expected.dtype.kind != 'b':
            return float('inf')
        return float(np.any(np.logical_xor(actual[mask], expected[mask])))
    return float(np.max(np.abs(actual[mask]-expected[mask]))) if np.any(mask) else 0.


def dense_fast_qualification(dense,fast_result,t,tolerance=1e-9):
    errors={}
    fields=sorted(set(key for key,value in dense.items() if isinstance(value,np.ndarray))|set(key for key,value in fast_result.items() if isinstance(value,np.ndarray)))
    for key in fields:
        a,b=dense.get(key),fast_result.get(key)
        if a is None or b is None:
            if a is not None or b is not None:raise RuntimeError('Dense/fast array/NA differs: '+key)
            continue
        errors[key]=difference(a,b)
    equal_iterations=int(dense['sweeps'])==int(fast_result['sweeps'])
    equal_cap=bool(not dense['converged'])==bool(not fast_result['converged'])
    passed=all(value<=tolerance for value in errors.values()) and equal_iterations and equal_cap
    receipt=dict(passed=passed,maximum_absolute_errors=errors,iterations_equal=equal_iterations,cap_equal=equal_cap,tolerance=tolerance)
    if not passed:raise RuntimeError('Dense/fast gate failed t='+str(t)+': '+json.dumps(receipt))
    return receipt


def v53_qualification(reference, outputs, baseline_states, atomic_results, t):
    errors = {}
    for name, key in (('old_support', 'single_support'), ('old_mean', 'single_mean'),
                       ('gaussian', 'gaussian'), ('hard_box', 'hard_box')):
        errors[name+'__theta'] = difference(outputs[name], reference[key+'__theta'][t])
    for key in ('q', 'logq', 'current_mean', 'current_cov'):
        errors['old__'+key] = difference(atomic_results['old'][key], reference['single__'+key][t])
    for name, state in baseline_states.items():
        for key, value in state.items():
            refkey = name+'__'+key
            if isinstance(value, np.ndarray) and refkey in reference and key != 'theta':
                errors[refkey] = difference(value, reference[refkey][t])
    passed = all(value <= 1e-9 for value in errors.values())
    if not passed:
        raise RuntimeError('Own V53 reference gate failed t='+str(t)+': '+json.dumps(errors))
    return dict(passed=passed, maximum_absolute_errors=errors, tolerance=1e-9,
                saved_states_used_only_after_private_updates=True)


def infer(n, observations, protocol, reference_path=None, capture=True):
    if set(observations) != {'states', 'inputs', 'a', 'eta', 'sigma'}:
        raise ValueError('Only the sanitized observation whitelist enters inference')
    started = time.perf_counter()
    obj, B = structure(n)
    initialization = dict(structure=time.perf_counter()-started,
        model_load={}, model_copies={}, actors={}, plans={}, spectra={})
    actors, plans, pairs = {}, {}, {}
    # Each constructor is charged its own actual model read and private copy.
    for name in FILTERS:
        if name in CHAINS:
            clock = time.perf_counter(); model = model_load(protocol)
            initialization['model_load'][name] = time.perf_counter()-clock
            clock = time.perf_counter(); private_model = (copy.deepcopy(model) if name == 'old'
                else kernel_actor.make_model(copy.deepcopy(model)))
            initialization['model_copies'][name] = time.perf_counter()-clock
        else:
            initialization['model_load'][name] = 0.
            initialization['model_copies'][name] = 0.
        clock = time.perf_counter()
        if name == 'old':
            actors[name] = OldActor(obj.edges, private_model, paired=False)
        elif name in ('dense', 'fast'):
            actors[name] = kernel_actor.Actor(obj.m, private_model, mode=name)
        elif name == 'gaussian':
            actors[name] = Gaussian(obj.m)
        else:
            actors[name] = Author(obj.m, kind='hard', tau=.1, box=True)
        initialization['actors'][name] = time.perf_counter()-clock
        clock = time.perf_counter(); plans[name] = fast.IncidenceStatisticsPlan(obj)
        initialization['plans'][name] = time.perf_counter()-clock
    for name in METHODS:
        clock = time.perf_counter(); pairs[name] = spectral.SpectrumPair(n, B)
        initialization['spectra'][name] = time.perf_counter()-clock
    standalone_initialization = {}
    for name in METHODS:
        owner = name.split('_')[0] if name.endswith(('_support', '_mean')) else name
        standalone_initialization[name] = initialization['structure']+sum(
            initialization[group][owner] for group in ('model_load', 'model_copies', 'actors', 'plans'))
        standalone_initialization[name] += initialization['spectra'][name]
    fixed, rows, snapshots = {}, [], {}
    points = {name: [] for name in METHODS}
    caps = {name: 0 for name in CHAINS}
    baseline_failures = spectral_failures = 0
    reference = None
    try:
        for t in range(T):
            c.tick()
            costs = dict(readouts={}, graph={}, spectrum={}, capture={}, reference_gate=0., fingerprint=0.)
            results, outputs, baseline_states, metas = {}, {}, {}, {}
            # Snapshot persistence is separate from inference cost and includes
            # the state immediately before the first t36 update.
            if capture and t == 36:
                clock = time.perf_counter()
                snapshots['before_t036'] = {name: actors[name].snapshot() for name in ('dense', 'fast')}
                costs['snapshot_capture'] = time.perf_counter()-clock
            order = FILTERS[t % len(FILTERS):]+FILTERS[:t % len(FILTERS)]
            for name in order:
                clock = time.perf_counter()
                stats, X, response = statistics_for(obj, plans[name], observations, t)
                costs[name+'_statistics'] = time.perf_counter()-clock
                clock = time.perf_counter()
                if name in CHAINS:
                    current = actors[name].update(stats.G, stats.h)
                    costs[name+'_update'] = time.perf_counter()-clock
                    results[name] = current
                    costs[name+'_temporal'] = float(current['temporal_seconds'])
                    costs[name+'_non_temporal'] = costs[name+'_update']-costs[name+'_temporal']
                    # Readouts were computed and copied within this paid update.
                    outputs[name+'_support'] = current['support_action']
                    outputs[name+'_mean'] = current['mean_action']
                    costs['readouts'][name+'_support'] = 0.
                    costs['readouts'][name+'_mean'] = 0.
                    caps[name] += int(not current['converged'])
                    clock = time.perf_counter()
                    metas[name] = scalar_meta(current)
                    metas[name].update(fingerprint_metadata(current))
                    if capture:
                        capture_atomic(fixed, name, current, t)
                    costs['capture'][name] = time.perf_counter()-clock
                else:
                    if name == 'hard_box':
                        point, meta = actors[name].update(stats, dense_loader=dense_loader(
                            obj, B, X, response, float(observations['eta']), float(observations['sigma'])),
                            compute_score=False)
                    else:
                        point, meta = actors[name].update(stats)
                    costs[name+'_update'] = time.perf_counter()-clock
                    outputs[name] = point
                    baseline_failures += int(not meta['success'])
                    metas[name] = meta
                    clock = time.perf_counter()
                    state = actors[name].audit_state(copy=True)
                    baseline_states[name] = state
                    metas[name].update(fingerprint_metadata(state))
                    if capture:
                        capture_raw_arrays(fixed,name,state,t,exclude=('theta',))
                    costs['capture'][name] = time.perf_counter()-clock
            clock = time.perf_counter()
            equivalence = dense_fast_qualification(results['dense'], results['fast'], t)
            # The reference NPZ is first opened only after actual own updates.
            if reference is None and reference_path is not None:
                keys = {'single_support__theta', 'single_mean__theta',
                    'gaussian__theta', 'hard_box__theta'}
                keys.update('single__'+key for key in ('q', 'logq', 'current_mean', 'current_cov'))
                keys.update(name+'__'+key for name, state in baseline_states.items()
                    for key, value in state.items() if isinstance(value, np.ndarray) and key != 'theta')
                with np.load(reference_path, allow_pickle=False) as reader:
                    reference = {key: np.array(reader[key], copy=True) for key in keys if key in reader.files}
            old_match = (v53_qualification(reference, outputs, baseline_states, results, t)
                         if reference is not None else None)
            costs['reference_gate'] = time.perf_counter()-clock
            clock = time.perf_counter()
            fingerprints = dict(atomic={name: array_fingerprint(results[name]) for name in CHAINS},
                current_q={name: hash_arrays({'q': results[name]['q']}) for name in CHAINS},
                baseline={name: array_fingerprint(baseline_states[name]) for name in BASELINES}, points={},
                atomic_decisions={name:decision_state(results[name]) for name in CHAINS})
            costs['fingerprint'] += time.perf_counter()-clock
            spectral_meta = {}
            point_order = METHODS[t % len(METHODS):]+METHODS[:t % len(METHODS)]
            for name in point_order:
                point = np.asarray(outputs[name], dtype=float)
                if (point.shape != (obj.m,) or not np.all(np.isfinite(point))
                        or np.any((point < 0.) | (point > 1.))):
                    raise FloatingPointError('Illegal candidate-edge point: '+name)
                clock = time.perf_counter()
                L = spectral.make_laplacian(B, point)
                costs['graph'][name] = time.perf_counter()-clock
                costs['spectrum'][name], spectral_meta[name] = {}, {}
                spectral_arrays = {'theta': point, 'L': L}
                for mode in MODES:
                    clock = time.perf_counter()
                    values, U, meta = spectrum_step(pairs[name], mode, L, point)
                    costs['spectrum'][name][mode] = time.perf_counter()-clock
                    spectral_failures += int(not meta['numerical_pass'])
                    spectral_meta[name][mode] = meta
                    spectral_arrays[mode+'__values'] = values
                    spectral_arrays[mode+'__U'] = U
                pairs[name].t += 1
                clock = time.perf_counter()
                points[name].append(point.copy())
                if capture:
                    for key, value in spectral_arrays.items():
                        append_fixed(fixed, name+'__'+key, value)
                costs['capture'][name+'_point'] = time.perf_counter()-clock
                clock = time.perf_counter()
                fingerprints['points'][name] = hash_arrays(spectral_arrays)
                costs['fingerprint'] += time.perf_counter()-clock
            costs['standalone'] = {}
            for name in METHODS:
                owner = name.split('_')[0] if name.endswith(('_support', '_mean')) else name
                state_cost = costs[owner+'_statistics']+costs[owner+'_update']
                costs['standalone'][name] = {mode: state_cost+costs['graph'][name]
                    +costs['spectrum'][name][mode] for mode in MODES}
            costs['joint_pipeline'] = sum(costs[name+'_statistics']+costs[name+'_update']
                for name in FILTERS)+sum(costs['graph'].values())+sum(
                sum(v.values()) for v in costs['spectrum'].values())
            rows.append(dict(t=t, filter_order=list(order), point_order=list(point_order),
                atomic_meta={name: metas[name] for name in CHAINS},
                baseline_meta={name: metas[name] for name in BASELINES}, costs=costs,
                spectra=spectral_meta, dense_fast_equivalence=equivalence, V53_reference_gate=old_match,
                output_fingerprints=fingerprints,
                model_contract=dict(node_observations_only=True, independent_private_histories=True,
                    fixed_model_npz=True, new_kernel_private_dense_and_fast=True,
                    baseline_tau_fixed=.1, truth_read=False)))
        if capture:
            clock = time.perf_counter()
            snapshots['end_t072'] = {name: actors[name].snapshot() for name in ('dense', 'fast')}
            rows[-1]['costs']['end_snapshot_capture'] = time.perf_counter()-clock
    finally:
        # reference is an own post-update qualification cache, never an actor input.
        reference = None
    arrays = stack_arrays(fixed) if capture else None
    return dict(arrays=arrays, steps=rows, snapshots=snapshots,
        point_sequences={name: np.stack(value) for name, value in points.items()},
        initialization_costs=initialization,
        standalone_initialization_seconds=standalone_initialization,
        joint_initialization_seconds=initialization['structure']+sum(
            sum(values.values()) for key, values in initialization.items() if key != 'structure'),
        variational_cap_blocks=caps, baseline_solver_failures=baseline_failures,
        spectral_failures=spectral_failures,
        dense_fast_equivalence_passed=all(row['dense_fast_equivalence']['passed'] for row in rows),
        V53_reference_passed=all(row['V53_reference_gate'] is None
            or row['V53_reference_gate']['passed'] for row in rows))


def save_snapshots(folder, snapshots):
    arrays, schema = {}, {}
    for label, states in snapshots.items():
        schema[label] = {}
        for name, state in states.items():
            schema[label][name] = flatten_snapshot(state, arrays, label+'__'+name)
    npz(folder/'SNAPSHOTS.npz', arrays)
    c.fresh(folder/'SNAPSHOTS_SCHEMA.json', schema)
    return dict(arrays_sha256=c.sha(folder/'SNAPSHOTS.npz'),
                schema_sha256=c.sha(folder/'SNAPSHOTS_SCHEMA.json'),
                labels=list(schema), no_pickle=True, complete_core_snapshot=True)


def summary_counts(count):
    return dict(streams=count, blocks=count*T, atomic_posteriors=count*T*3,
        noninitial_pair_moment_vectors=count*(T-1)*3,
        baseline_snapshots=count*T*2, point_outputs=count*T*len(METHODS),
        full_point_spectra=count*T*len(METHODS)*len(MODES))


def run_data(stage, out, protocol):
    cases = (protocol['streams'] if stage == 'formal' else [case
        for case in protocol['streams'] if case['id'] in protocol['smoke_stream_ids']])
    model = model_load(protocol)
    npz(out/'MODEL_old.npz', model)
    npz(out/'MODEL_new.npz', {key: np.asarray(value) for key,value in kernel_actor.make_model(copy.deepcopy(model)).items()})
    streams = []
    for case in cases:
        c.tick()
        folder = out/'streams'/case['id']
        folder.mkdir(parents=True, exist_ok=False)
        observations = sanitize(case, folder/'observations.npz')
        result = infer(case['n'], observations, protocol,
            reference_path=source_path(case['reference_inference']), capture=True)
        npz(folder/'INFERENCE.npz', result['arrays'])
        snapshot_receipt = save_snapshots(folder, result['snapshots'])
        meta = dict(case_id=case['id'], n=case['n'], T=T, methods=list(METHODS),
            chains=list(CHAINS), baseline_chains=list(BASELINES), spectral_modes=list(MODES),
            source_observations_sha256=c.sha(source_path(case['source_observations'])),
            observations_sha256=c.sha(folder/'observations.npz'),
            inference_sha256=c.sha(folder/'INFERENCE.npz'), model_source=protocol['model_source'],
            model_source_sha256=c.sha(source_path(protocol['model_source'])),
            snapshots=snapshot_receipt, steps=result['steps'],
            initialization_costs=result['initialization_costs'],
            standalone_initialization_seconds=result['standalone_initialization_seconds'],
            joint_initialization_seconds=result['joint_initialization_seconds'],
            inference_completed_before_truth_evaluation=True, truth_members_read=False,
            variational_cap_blocks=result['variational_cap_blocks'],
            baseline_solver_failures=result['baseline_solver_failures'],
            spectral_failures=result['spectral_failures'],
            dense_fast_equivalence_passed=result['dense_fast_equivalence_passed'],
            V53_reference_passed=result['V53_reference_passed'],
            storage_schema=dict(fixed_current_stacks=T,temporal_first_block='numeric_NaN',
                all_raw_ndarrays_saved=True,scalar_None_never_array_collected=True,
                raw_fingerprint_fields_and_dtypes_in_step_meta=True,
                adjacent_pair_history_saved=False,bounded_snapshots=True,
                capture_excluded_from_algorithm_cost=True),
            cost_contract=dict(two_atomic_readouts_share_state=True,
                point_cost='own_state_plus_own_graph_plus_one_spectral_route_plus_initialization',
                joint_pipeline_counted_separately=True, update_includes_all_state_copies=True,
                prediction_and_temporal_charged=True,
                qualification_fingerprinting_capture_persistence_separate=True))
        c.fresh(folder/'META.json', meta)
        receipt = {key: result[key] for key in ('variational_cap_blocks',
            'baseline_solver_failures', 'spectral_failures',
            'dense_fast_equivalence_passed', 'V53_reference_passed')}
        streams.append(dict(id=case['id'], path='streams/'+case['id'], **receipt))
        print(json.dumps(dict(stage=stage, completed_stream=case['id'], **receipt)), flush=True)
    return dict(status='completed', phase=stage, streams=streams,
        counts=summary_counts(len(streams)), variational_cap_blocks={name:
            sum(row['variational_cap_blocks'][name] for row in streams) for name in CHAINS},
        baseline_solver_failures=sum(row['baseline_solver_failures'] for row in streams),
        spectral_failures=sum(row['spectral_failures'] for row in streams),
        integrity_passed=all(row['baseline_solver_failures'] == row['spectral_failures'] == 0
            and row['dense_fast_equivalence_passed'] and row['V53_reference_passed'] for row in streams),
        caps_retained=True, no_truth_members_read_by_inference=True, independent_confirmation=False)


def replay(out, protocol):
    selected = [case for case in protocol['streams'] if case['id'] in protocol['replay_stream_ids']]
    receipts = []
    for repeat in range(int(protocol['replay_repeats'])):
        for case in selected:
            c.tick()
            result = infer(case['n'], sanitize(case), protocol,
                reference_path=source_path(case['reference_inference']), capture=False)
            formal = ROOT/'runs/formal/streams'/case['id']
            formal_meta = c.read(formal/'META.json')
            matches = [row['output_fingerprints'] == expected['output_fingerprints']
                for row, expected in zip(result['steps'], formal_meta['steps'])]
            with np.load(formal/'INFERENCE.npz', allow_pickle=False) as raw:
                points_match = {name: bool(np.array_equal(value, raw[name+'__theta']))
                    for name, value in result['point_sequences'].items()}
            if (len(matches) != T or not all(matches) or not all(points_match.values())
                    or result['baseline_solver_failures'] or result['spectral_failures']):
                raise RuntimeError('Private same-history replay gate failed')
            receipts.append(dict(repeat=repeat, stream=case['id'],
                all_output_and_state_fingerprints_match=True,
                blocks_fingerprints_match=matches, point_sequences_match=points_match,
                steps=result['steps'], initialization_costs=result['initialization_costs'],
                standalone_initialization_seconds=result['standalone_initialization_seconds'],
                joint_initialization_seconds=result['joint_initialization_seconds'],
                variational_cap_blocks=result['variational_cap_blocks'],
                baseline_solver_failures=result['baseline_solver_failures'],
                spectral_failures=result['spectral_failures'],
                real_private_histories=True, scientifically_new=False, capture=False))
            print(json.dumps(dict(stage='replay', repeat=repeat, completed_stream=case['id'])), flush=True)
    c.fresh(out/'REPLAYS.json', dict(repeats=protocol['replay_repeats'],
        streams=protocol['replay_stream_ids'], receipts=receipts,
        private_statistics_and_filter_histories=True, scientifically_new=False))
    counts = summary_counts(len(receipts)); counts['histories'] = counts.pop('streams')
    return dict(status='completed', integrity_passed=True, counts=counts,
        all_points_and_state_fingerprints_reproduced=True,
        variational_cap_blocks={name: sum(row['variational_cap_blocks'][name]
            for row in receipts) for name in CHAINS},
        baseline_solver_failures=0, spectral_failures=0, scientifically_new=False)


def checkpoint(out, protocol):
    """Restore real formal core states and independently recompute all tails.

    This is separately paid qualification. No graph/spectrum or truth is used,
    and none of its timings enters the formal point algorithm-cost comparison.
    """
    before = int(protocol['checkpoint_before_t'])
    selected = [case for case in protocol['streams']
                if case['id'] in protocol['checkpoint_stream_ids']]
    if before != 36 or len(selected) != 28:
        raise ValueError('Fixed all-28, pre-t36 checkpoint population required')
    receipts=[]
    for case in selected:
        c.tick()
        formal=ROOT/'runs/formal/streams'/case['id']
        meta=c.read(formal/'META.json')
        schema_path, arrays_path = formal/'SNAPSHOTS_SCHEMA.json', formal/'SNAPSHOTS.npz'
        if (c.sha(schema_path)!=meta['snapshots']['schema_sha256']
                or c.sha(arrays_path)!=meta['snapshots']['arrays_sha256']):
            raise RuntimeError('Formal snapshot binding failed')
        schema=c.read(schema_path)
        observations=sanitize(case)
        for name in ('dense','fast'):
            costs={}
            clock=time.perf_counter();obj,B=structure(case['n'])
            costs['structure']=time.perf_counter()-clock
            clock=time.perf_counter();model=model_load(protocol)
            costs['model_load']=time.perf_counter()-clock
            clock=time.perf_counter();private_model=kernel_actor.make_model(copy.deepcopy(model))
            costs['model_copy']=time.perf_counter()-clock
            clock=time.perf_counter();actor=kernel_actor.Actor(obj.m,private_model,mode=name)
            costs['actor_init']=time.perf_counter()-clock
            clock=time.perf_counter();plan=fast.IncidenceStatisticsPlan(obj)
            costs['plan_init']=time.perf_counter()-clock
            clock=time.perf_counter()
            with np.load(arrays_path,allow_pickle=False) as raw:
                state=unflatten_snapshot(schema['before_t036'][name],raw)
            actor.restore(state)
            costs['restore']=time.perf_counter()-clock
            if int(actor.step)!=before:
                raise RuntimeError('Actual restored checkpoint step differs')
            costs.update(statistics=0.,updates=0.,qualification=0.)
            steps=[]
            for t in range(before,T):
                c.tick();clock=time.perf_counter()
                stats,X,response=statistics_for(obj,plan,observations,t)
                statistics_seconds=time.perf_counter()-clock
                clock=time.perf_counter();current=actor.update(stats.G,stats.h)
                update_seconds=time.perf_counter()-clock
                clock=time.perf_counter()
                atomic_fingerprint=array_fingerprint(current)
                q_fingerprint=hash_arrays({'q':current['q']})
                decisions=decision_state(current)
                expected=meta['steps'][t]['output_fingerprints']
                expected_atomic=expected['atomic'][name]
                expected_q=expected['current_q'][name]
                expected_decisions=expected['atomic_decisions'][name]
                matched=(atomic_fingerprint==expected_atomic and q_fingerprint==expected_q
                         and decisions==expected_decisions)
                qualification_seconds=time.perf_counter()-clock
                steps.append(dict(t=t,statistics_seconds=statistics_seconds,
                    update_seconds=update_seconds,qualification_seconds=qualification_seconds,
                    atomic_fingerprint=atomic_fingerprint,
                    expected_atomic_fingerprint=expected_atomic,
                    q_fingerprint=q_fingerprint,expected_q_fingerprint=expected_q,
                    decisions=decisions,expected_decisions=expected_decisions,match=matched))
                costs['statistics']+=statistics_seconds
                costs['updates']+=update_seconds
                costs['qualification']+=qualification_seconds
                if not matched:
                    raise RuntimeError('Actual formal checkpoint tail differs: '+case['id']+' '+name+' t='+str(t))
            receipts.append(dict(stream=case['id'],chain=name,before_t=before,
                tail_blocks=T-before,snapshot_label='before_t036',snapshot_step=int(state['step']),
                snapshot_schema_sha256=c.sha(schema_path),snapshot_arrays_sha256=c.sha(arrays_path),
                steps=steps,costs=costs,all_fingerprints_match=True,
                cost_scope='Actual restored private statistics and updates; qualification only, excluded from formal algorithm timing'))
        print(json.dumps(dict(stage='checkpoint',completed_stream=case['id'])),flush=True)
    c.fresh(out/'CHECKPOINTS.json',dict(before_t=before,
        streams=protocol['checkpoint_stream_ids'],receipts=receipts,
        private_statistics=True,original_formal_snapshots=True,qualifies_main_timing=False))
    return dict(status='completed',integrity_passed=True,
        counts=dict(streams=len(selected),restored_histories=len(receipts),
            tail_blocks_per_history=T-before,restored_blocks=len(receipts)*(T-before),
            atomic_posteriors=len(receipts)*(T-before),full_point_spectra=0),
        all_core_states_and_outputs_reproduced=True,
        qualification_cost_seconds=sum(sum(row['costs'].values()) for row in receipts),
        scientifically_new=False)


def main(stage, output):
    admission = c.verify()
    c.tick()
    out = Path(output).resolve()
    if ROOT.resolve() not in out.parents:
        raise ValueError('Worker output must be a new directory below this V60 root')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        runtime = c.runtime()
        load_numeric()
        protocol = c.read(ROOT/'PROTOCOL_V60.json')
        with limits(limits=1):
            import platform
            import numba
            numba.set_num_threads(1)
            c.fresh(out/'ENVIRONMENT.json', dict(python=sys.version,
                platform=platform.platform(), runtime=runtime,
                thread_environment={key: os.environ.get(key) for key in (
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')},
                numba_threads=numba.get_num_threads(), admission=admission,
                numerical_import_after_admission=True))
            before = pools()
            if not before or any(pool['num_threads'] != 1 for pool in before):
                raise RuntimeError('Actual native pools are not single threaded')
            c.fresh(out/'THREADPOOLS_BEFORE.json', before)
            c.fresh(out/'SETUP.json', setup(protocol))
            initial_signatures = signatures()
            summary = (replay(out, protocol) if stage == 'replay' else
                checkpoint(out, protocol) if stage == 'checkpoint' else run_data(stage, out, protocol))
            after = pools()
            summary.update(wall_seconds=time.perf_counter()-started,
                signature_unchanged=initial_signatures == signatures(),
                all_native_pools_single_thread=bool(after) and all(p['num_threads'] == 1 for p in after),
                numba_single_thread=numba.get_num_threads() == 1)
            summary['timing_qualified'] = (summary['signature_unchanged']
                and summary['all_native_pools_single_thread'] and summary['numba_single_thread'])
            c.fresh(out/'THREADPOOLS_AFTER.json', after)
            c.fresh(out/'SIGNATURES_AFTER.json', signatures())
            summary['final_verification'] = c.verify()
            c.fresh(out/'SUMMARY.json', summary)
            print(json.dumps({key: value for key, value in summary.items() if key != 'streams'}), flush=True)
            return 0 if summary['integrity_passed'] and summary['timing_qualified'] else 1
    except BaseException:
        c.fresh(out/'FAILURE.json', dict(error=traceback.format_exc(),
            wall_seconds=time.perf_counter()-started, single_attempt_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('smoke', 'formal', 'checkpoint', 'replay'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.stage, args.output))

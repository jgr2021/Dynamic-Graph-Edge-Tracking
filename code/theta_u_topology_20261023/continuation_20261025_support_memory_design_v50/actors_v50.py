"""Private frozen-V46 actors for one support-transition intervention only.

Numerical imports/definitions are loaded lazily after the parent's admission.
Both variants use the same frozen V47 temporal implementation and original V46
prediction/CAVI; supportneutral changes support while retaining positive row laws.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V46_SOURCE = PROJECT/'continuation_20261025_atomic_nodes_v46/atomic_nodes.py'
V46_SHA256 = 'cccedcdc135d8ee4ca4b088ceca66b3f347d762af57e785cd13843d70659a30f'
KERNEL_SOURCE = PROJECT/'continuation_20261025_temporal_kernel_design_v47/temporal_kernel.py'
KERNEL_SHA256 = '26b975ef2789f7bdda3d5f8729f9064b426d5c632a111e691a4e24ca89568f3f'
VARIANTS = ('markov', 'supportneutral')
ALIASES = dict(markov='_theta_atomic_v46_markov_v50',
               supportneutral='_theta_atomic_v46_supportneutral_v50')
KERNEL_ALIAS = '_theta_temporal_kernel_v47_v50'
TEMPORAL_KEYS = ('old_smoothed_mean', 'old_smoothed_variance', 'cross_diag',
    'delta_mean', 'delta_variance', 'temporal_current_marginal_error',
    'temporal_conditional_normalization_error', 'temporal_delta_formula_error',
    'temporal_covariance_determinant_minimum')
_LOADED = None
_SETUP_RECEIPT = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _variant(variant):
    if variant not in VARIANTS:
        raise ValueError('Variant must be markov or supportneutral')


def _load_file(alias, source):
    if alias in sys.modules:
        raise ImportError('Private V50 alias already exists: '+alias)
    spec = importlib.util.spec_from_file_location(alias, source)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load admitted frozen definitions')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source.resolve():
        raise ImportError('Actual private alias source differs')
    return module


def _timed_function(target, timer):
    def timed(logold, logtransition, logprior, q, values):
        started = time.perf_counter()
        try:
            return target(logold, logtransition, logprior, q, values)
        finally:
            timer['seconds'] += time.perf_counter()-started
            timer['calls'] += 1
    return timed


def _load():
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if _sha(V46_SOURCE) != V46_SHA256 or _sha(KERNEL_SOURCE) != KERNEL_SHA256:
        raise RuntimeError('Frozen V46 actor or V47 temporal source differs')
    modules = {variant: _load_file(ALIASES[variant], V46_SOURCE) for variant in VARIANTS}
    kernel = _load_file(KERNEL_ALIAS, KERNEL_SOURCE)
    timers = {variant: dict(seconds=0., calls=0, busy=False) for variant in VARIANTS}
    wrappers = {}
    for variant, module in modules.items():
        wrappers[variant] = _timed_function(kernel.temporal_moments, timers[variant])
        module.temporal_moments = wrappers[variant]
    _LOADED = dict(modules=modules, kernel=kernel, timers=timers, wrappers=wrappers)
    return _LOADED


def _require_loaded():
    if _LOADED is None:
        raise RuntimeError('Call load_modules/make_actor/setup after admission first')
    return _LOADED


def load_modules():
    return dict(_load()['modules'])


def _supportneutral_model(model, np):
    result = {key: value.copy() if hasattr(value, 'copy') else value
              for key, value in model.items()}
    k = len(result['values'])
    # Retain finite logarithmic positive tails without an exp-to-log roundtrip.
    module = _require_loaded()['modules']['supportneutral']
    positive = result['logtransition'][:, 1:]
    normalizer = module.logsumexp(positive, axis=1)
    if not np.all(np.isfinite(normalizer)):
        raise ValueError('Every row requires reachable positive support')
    loghalf = np.log(.5)
    result['logtransition'] = np.empty((k, k), dtype=float)
    result['logtransition'][:, 0] = loghalf
    result['logtransition'][:, 1:] = loghalf+positive-normalizer[:, None]
    result['transition'] = np.exp(result['logtransition'])
    result['transition'][:, 0] = .5
    return result


class _ActorProxy:
    def __init__(self, actor, variant, timer):
        self._actor, self.variant, self._timer = actor, variant, timer

    @property
    def m(self):
        return self._actor.m

    @property
    def step(self):
        return self._actor.step

    def __getattr__(self, name):
        return getattr(self._actor, name)

    def model_receipt(self):
        return self._actor.model_receipt()

    def update(self, G, h):
        if self._timer['busy']:
            raise RuntimeError('Overlapping updates of one V50 variant are forbidden')
        self._timer.update(seconds=0., calls=0, busy=True)
        try:
            result = self._actor.update(G, h)
        finally:
            self._timer['busy'] = False
        result['temporal_seconds'] = float(self._timer['seconds'])
        result['temporal_calls'] = int(self._timer['calls'])
        return result


def make_actor(m, variant, values=None, transition=None, initial=None):
    _variant(variant)
    loaded = _load()
    module = loaded['modules'][variant]
    actor = module.Actor(m, values=values, transition=transition, initial=initial)
    if variant == 'supportneutral':
        # Only a newly constructed private actor is changed, before any update.
        # Keep all grid/prior metadata; do not patch prediction, CAVI or q.
        actor._model = _supportneutral_model(actor._model, module.np)
        actor.transition = actor._model['transition']
        actor.logtransition = actor._model['logtransition']
    return _ActorProxy(actor, variant, loaded['timers'][variant])


def get_namespace(variant):
    _variant(variant)
    return _require_loaded()['modules'][variant]


def model_receipt(variant):
    _variant(variant)
    module = _require_loaded()['modules'][variant]
    model = module.default_model()
    return _supportneutral_model(model, module.np) if variant == 'supportneutral' else model


def signature_inventory():
    if _LOADED is None:
        return dict(loaded=False, markov={}, supportneutral={}, kernel={})
    return dict(loaded=True,
        markov=_LOADED['modules']['markov'].signature_inventory(),
        supportneutral=_LOADED['modules']['supportneutral'].signature_inventory(),
        kernel=_LOADED['kernel'].signature_inventory())


def binding_receipt():
    loaded = _require_loaded()
    if _sha(V46_SOURCE) != V46_SHA256 or _sha(KERNEL_SOURCE) != KERNEL_SHA256:
        raise RuntimeError('Frozen numerical source changed after loading')
    np = loaded['modules']['markov'].np
    models = {variant: model_receipt(variant) for variant in VARIANTS}
    left, right = (loaded['modules'][variant] for variant in VARIANTS)
    shared_cache = left._DEFAULT_MODEL_CACHE is not None and left._DEFAULT_MODEL_CACHE is right._DEFAULT_MODEL_CACHE
    if left is right or left.__dict__ is right.__dict__ or shared_cache:
        raise RuntimeError('Private V50 namespaces or default caches are shared')
    variants = {}
    for variant in VARIANTS:
        module = loaded['modules'][variant]
        if module.temporal_moments is not loaded['wrappers'][variant]:
            raise RuntimeError('Common temporal implementation binding changed')
        variants[variant] = dict(alias=module.__name__, file=str(Path(module.__file__).resolve()),
            source_sha256=V46_SHA256, actor_class_module=module.Actor.__module__,
            temporal_target_module=loaded['kernel'].__name__,
            temporal_target_source=str(KERNEL_SOURCE.resolve()), temporal_target_sha256=KERNEL_SHA256,
            temporal_target_is_validated_v47_kernel=True,
            transition_policy='frozen_v46_markov' if variant == 'markov' else 'half_support_preserve_positive_conditional')
    same = all(np.array_equal(models['markov'][key], models['supportneutral'][key])
               for key in ('values', 'initial', 'loginitial', 'positive_bin_edges',
                           'birth_probabilities', 'birth_log_probabilities'))
    supportneutral = models['supportneutral']
    original = models['markov']
    positive = supportneutral['logtransition'][:, 1:]
    original_positive = original['logtransition'][:, 1:]
    lse = loaded['modules']['supportneutral'].logsumexp
    conditional = positive-lse(positive, axis=1)[:, None]
    original_conditional = original_positive-lse(original_positive, axis=1)[:, None]
    row_law_preserved = np.allclose(conditional, original_conditional, rtol=0., atol=1e-12)
    zero_half = np.all(supportneutral['transition'][:, 0] == .5)
    logzero_half = np.all(supportneutral['logtransition'][:, 0] == np.log(.5))
    if not same or not row_law_preserved or not zero_half or not logzero_half:
        raise RuntimeError('Single support-transition intervention contract differs')
    return dict(variants=variants, distinct_namespaces=True, default_model_cache_shared=False,
        source_file_same=True, source_sha256=V46_SHA256,
        common_temporal_kernel=True, temporal_kernel_sha256=KERNEL_SHA256,
        common_timing_wrapper_factory=True, only_support_transition_intervention=True,
        same_values_and_initial=True, row_positive_conditional_law_preserved=True,
        supportneutral_zero_transition_probability_is_half=True,
        supportneutral_zero_transition_logprobability_is_log_half=True,
        supportneutral_prediction_shortcut=False,
        original_V46_file_edited=False, original_V47_file_edited=False,
        prediction_and_CAVI_source_unchanged=True, point_readout_rule_unchanged=True,
        initialization_from_saved_posterior=False)



def setup():
    global _SETUP_RECEIPT
    if _SETUP_RECEIPT is not None:
        return dict(_SETUP_RECEIPT, already_warmed=True)
    started = time.perf_counter()
    loaded = _load()
    np = loaded['modules']['markov'].np
    component = dict(markov=loaded['modules']['markov'].setup(),
                     supportneutral=loaded['modules']['supportneutral'].setup(),
                     kernel=loaded['kernel'].setup())
    G = np.diag(np.asarray([2., 4., 6.]))
    h = np.asarray([.1, .8, 1.1])
    steps = {}
    for variant in VARIANTS:
        actor = make_actor(3, variant)
        history = []
        for t in range(2):
            result = actor.update(G, h)
            if result['temporal_calls'] != int(t > 0) or not result['converged']:
                raise RuntimeError('Artificial V50 warmup qualification failed')
            history.append(dict(t=t, temporal_calls=result['temporal_calls'],
                                converged=result['converged']))
        steps[variant] = history
    _SETUP_RECEIPT = dict(artificial_only=True, scientific_observations_used=False,
        RNG_used=False, m=3, K=65, steps_per_variant=2, component_receipts=component,
        artificial_steps=steps, module_bindings=binding_receipt(),
        signatures=signature_inventory(), wall_seconds=float(time.perf_counter()-started))
    return dict(_SETUP_RECEIPT, already_warmed=False)

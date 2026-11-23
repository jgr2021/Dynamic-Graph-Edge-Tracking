"""Admitted frozen Markov control and new singleton/pair actors."""
from pathlib import Path
import importlib.util
import itertools
import math
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OLD_SOURCE=PROJECT/'continuation_20261025_support_memory_design_v50/actors_v50.py'
OLD=None
NEW=None


def load_modules():
    global OLD,NEW
    if OLD is None:
        alias='_frozen_actors_v50_control_for_v52'
        if alias in sys.modules:raise ImportError('Private control alias already loaded')
        spec=importlib.util.spec_from_file_location(alias,OLD_SOURCE)
        OLD=importlib.util.module_from_spec(spec);sys.modules[alias]=OLD;spec.loader.exec_module(OLD)
        OLD.load_modules()
        import pair_actor_v52
        if Path(pair_actor_v52.__file__).resolve()!=(ROOT/'pair_actor_v52.py').resolve():
            raise ImportError('Mixed pair actor source')
        NEW=pair_actor_v52
    return dict(original=OLD,paired=NEW)


def make_actor(m,variant):
    load_modules()
    if variant=='markov':return OLD.make_actor(m,'markov')
    if variant not in ('single','pair'):raise ValueError('Unknown actor variant')
    import numpy as np
    n=int((1+math.isqrt(1+8*m))//2)
    if n*(n-1)//2!=m:raise ValueError('Complete public candidate-edge dimension required')
    edges=np.asarray(list(itertools.combinations(range(n),2)),dtype=np.int64)
    return NEW.Actor(edges,OLD.model_receipt('markov'),paired=variant=='pair')


def model_receipt(variant='markov'):
    load_modules()
    return OLD.model_receipt('markov')


def setup():
    load_modules()
    return dict(original=OLD.setup(),new_engine=NEW.setup())


def signature_inventory():
    load_modules()
    return dict(original=OLD.signature_inventory(),pair_engine=NEW.signature_inventory())


def binding_receipt():
    load_modules()
    return dict(original=OLD.binding_receipt(),new_source=str(NEW.__file__),
                same_original_model=True,independent_private_actor_histories=True)

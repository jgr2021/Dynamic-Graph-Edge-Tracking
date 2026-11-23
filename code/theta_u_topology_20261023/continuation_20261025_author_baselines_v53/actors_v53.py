"""Frozen same-model singleton control; no author code is imported."""
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OLD=None


def load_modules():
    global OLD
    if OLD is None:
        source=PROJECT/'continuation_20261025_pair_recursive_v52/actors_v52.py'
        sys.path.insert(0,str(source.parent))
        spec=importlib.util.spec_from_file_location('_frozen_single_control_for_v53',source)
        OLD=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=OLD
        spec.loader.exec_module(OLD)
        OLD.load_modules()
    return OLD


def make_actor(m):
    return load_modules().make_actor(m,'single')


def model_receipt():
    return load_modules().model_receipt('markov')


def setup():
    return load_modules().setup()


def signature_inventory():
    return load_modules().signature_inventory()


def binding_receipt():
    return load_modules().binding_receipt()

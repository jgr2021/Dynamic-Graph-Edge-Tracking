"""Cached incidence expansion of the unchanged V16 NodeStatistics.

Import only defines interfaces and checks actual imported module origins. It
does not construct a plan, call a kernel, compile, generate data, or run setup.
Only the current node-to-dense Gram expansion is replaced; no estimator,
history, support, objective, stopping rule, or complete spectrum is changed.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit


ROOT = Path(__file__).resolve().parent
V16 = ROOT.parent / "continuation_20261024_fixed_star_development_v16"
if str(V16) not in sys.path:
    sys.path.insert(0, str(V16))
_node = importlib.import_module("node_updates")
_structured = importlib.import_module("structured_stats")
for _module, _filename in ((_node, "node_updates.py"), (_structured, "structured_stats.py")):
    if Path(_module.__file__).resolve() != (V16 / _filename).resolve():
        raise ImportError("V21 requires the original single V16 library: "+_filename)

NodeStatistics = _node.NodeStatistics
BlockStatistics = _structured.BlockStatistics
EdgeObservationStructure = _structured.EdgeObservationStructure
_SETUP = None


@njit(cache=False, fastmath=False)
def _expand_incident_gram(C, edges, offsets, incident_ids, incident_signs, G):
    """Same vertex/incident-edge/incident-edge order as the original scatter.

    Every ordered pair in each incident block is included. Shared diagonal
    contributions remain two separately added endpoint contributions. Dense
    initialization, scaling and final NumPy symmetrization are caller costs.
    """
    for vertex in range(len(offsets)-1):
        for first in range(offsets[vertex], offsets[vertex+1]):
            e = incident_ids[first]
            i, j = edges[e, 0], edges[e, 1]
            sign_e = incident_signs[first]
            for second in range(offsets[vertex], offsets[vertex+1]):
                f = incident_ids[second]
                u, v = edges[f, 0], edges[f, 1]
                covariance = ((C[i, u]-C[i, v])-C[j, u])+C[j, v]
                contribution = (sign_e*incident_signs[second])*covariance
                G[e, f] += contribution


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    checksum = hashlib.sha256()
    checksum.update(str(array.dtype).encode("ascii"))
    checksum.update(str(array.shape).encode("ascii"))
    checksum.update(array.tobytes(order="C"))
    return checksum.hexdigest()


class IncidenceStatisticsPlan:
    """A private, observed-support-independent CSR cache for one pipeline.

    A new plan must be constructed when n, candidate endpoints, endpoint order
    or orientation changes. No time-varying weight/support, state, G, h, or
    another pipeline's data are cached. Each candidate edge remains included.
    Construction cost and cache bytes are recorded rather than made free.
    """

    def __init__(self, structure):
        started = time.perf_counter()
        if not isinstance(structure, EdgeObservationStructure):
            raise TypeError("Original V16 EdgeObservationStructure required")
        self.structure = structure
        self.n, self.m = int(structure.n), int(structure.m)
        self.edges = np.ascontiguousarray(structure.edges, dtype=np.int64).copy()
        self.offsets = np.empty(self.n+1, dtype=np.int64)
        self.offsets[0] = 0
        ids_parts, signs_parts = [], []
        if len(structure.incident) != self.n:
            raise ValueError("Candidate incidence must contain each node once")
        for vertex, (raw_ids, raw_signs) in enumerate(structure.incident):
            ids = np.ascontiguousarray(raw_ids, dtype=np.int64)
            signs = np.ascontiguousarray(raw_signs, dtype=np.float64)
            if ids.ndim != 1 or signs.shape != ids.shape:
                raise ValueError("Invalid candidate incident-block shape")
            if np.any(ids < 0) or np.any(ids >= self.m):
                raise ValueError("Candidate incident index outside the fixed structure")
            expected_ids = np.flatnonzero(np.any(self.edges == vertex, axis=1))
            expected_signs = np.where(self.edges[ids, 0] == vertex, 1., -1.)
            if not np.array_equal(ids, expected_ids) or not np.array_equal(signs, expected_signs):
                raise ValueError("Candidate incidence order/orientation differs from the original structure")
            ids_parts.append(ids); signs_parts.append(signs)
            self.offsets[vertex+1] = self.offsets[vertex]+len(ids)
        self.incident_ids = np.ascontiguousarray(np.concatenate(ids_parts), dtype=np.int64)
        self.incident_signs = np.ascontiguousarray(np.concatenate(signs_parts), dtype=np.float64)
        if (len(self.incident_ids) != 2*self.m or
                not np.array_equal(np.bincount(self.incident_ids, minlength=self.m), np.full(self.m, 2))):
            raise ValueError("Each undirected candidate must occur at both endpoints")
        for value in (self.edges, self.offsets, self.incident_ids, self.incident_signs):
            value.setflags(write=False)
        self.array_hashes = {key: _array_sha256(getattr(self, key))
                             for key in ("edges", "offsets", "incident_ids", "incident_signs")}
        self.cache_bytes = int(sum(value.nbytes for value in
            (self.edges, self.offsets, self.incident_ids, self.incident_signs)))
        self.initialization_seconds = float(time.perf_counter()-started)

    def expand(self, stats):
        """Own paid replacement for stats.to_block_statistics(), returning G/h.

        Call the original NodeStatistics.from_observations first and charge it
        to this same pipeline. This function does not reconstruct or share C,
        D, h, Gdiag, observed arrays or another method's full Gram matrix.
        """
        if not isinstance(stats, NodeStatistics) or stats.structure is not self.structure:
            raise TypeError("This private plan requires own statistics on the identical fixed structure")
        if (self.structure.n != self.n or self.structure.m != self.m or
                not np.array_equal(self.structure.edges, self.edges)):
            raise ValueError("Candidate endpoints/order/orientation changed; construct a new paid plan")
        if stats.C.shape != (self.n, self.n):
            raise ValueError("Current node covariance dimension differs from the fixed plan")
        G = np.zeros((self.m, self.m))
        _expand_incident_gram(stats.C, self.edges, self.offsets,
                              self.incident_ids, self.incident_signs, G)
        # Keep original NumPy scaling, final symmetry, h copy and exact class.
        G *= stats.eta**2/stats.sigma**2
        G = (G+G.T)*0.5
        return BlockStatistics(stats.structure, G, stats.h.copy(), stats.response_norm_squared_scaled,
                               stats.centered_states, stats.responses, stats.eta, stats.sigma)

    def audit_state(self, copy=True):
        state = dict(n=self.n, m=self.m, edges=self.edges, offsets=self.offsets,
                     incident_ids=self.incident_ids, incident_signs=self.incident_signs,
                     initialization_seconds=self.initialization_seconds, cache_bytes=self.cache_bytes)
        return {key: value.copy() if copy and isinstance(value, np.ndarray) else value
                for key, value in state.items()}

    def audit_receipt(self):
        return dict(n=self.n, m=self.m, initialization_seconds=self.initialization_seconds,
            cache_bytes=self.cache_bytes, array_sha256=dict(self.array_hashes),
            candidate_edge_count=self.m, packed_endpoint_count=len(self.incident_ids),
            includes_all_candidates=True, candidate_order_and_orientation_preserved=True,
            private_plan=True, dynamic_support_cached=False, observed_statistics_cached=False,
            previous_G_or_h_reused=False, no_truth_support=True, no_oracle=True,
            initialization_is_separate_paid_pipeline_cost=True,
            dense_output_initialization_and_symmetry_paid_per_expand=True,
            node_statistics_source=str(Path(_node.__file__).resolve()),
            block_statistics_source=str(Path(_structured.__file__).resolve()),
            expansion_arithmetic="O(sum_v degree_v_squared + m_squared)",
            output_storage="O(m_squared)", cache_storage="O(n+m)",
            no_full_pipeline_complexity_or_speed_claim=True)


def signature_inventory():
    """Inventory only; never triggers compilation or a production update."""
    return {__name__+"._expand_incident_gram": [str(signature) for signature in _expand_incident_gram.signatures]}


def setup():
    """Explicit deterministic artificial production path; never run on import.

    This is signature setup plus an exact check against the old expansion on
    one fixed artificial input. It is not a scientific trajectory or timing
    benchmark, does not select parameters, and must be paid in the parent wall.
    """
    global _SETUP
    if _SETUP is not None:
        return dict(_SETUP, already_compiled=True)
    started = time.perf_counter()
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    structure = EdgeObservationStructure(4, edges)
    row, node = np.arange(1., 13.)[:, None], np.arange(1., 5.)[None, :]
    X = np.ascontiguousarray(np.sin(.17*row*node)+np.cos(.23*(row+1)*node))
    target = np.full(structure.m, .35); target[1] = 0.
    response = np.ascontiguousarray(.2*X@structure.laplacian(target))
    own = NodeStatistics.from_observations(structure, X, response, .2, .03)
    plan = IncidenceStatisticsPlan(structure)
    first_started = time.perf_counter()
    expanded = plan.expand(own)
    first_seconds = float(time.perf_counter()-first_started)
    original = own.to_block_statistics()
    exact_G = bool(np.array_equal(expanded.G, original.G))
    exact_h = bool(np.array_equal(expanded.h, original.h))
    if not (exact_G and exact_h):
        raise FloatingPointError("Fixed artificial incidence setup differs from original expansion; preserve failure, no rescue")
    _SETUP = dict(setup_seconds=float(time.perf_counter()-started),
        first_expand_including_compilation_seconds=first_seconds, plan=plan.audit_receipt(),
        artificial_n=4, artificial_m=6, artificial_rows=12, exact_G=exact_G, exact_h=exact_h,
        single_original_BlockStatistics_type=isinstance(expanded, BlockStatistics),
        node_statistics_unchanged=True, original_dense_expansion_only_reference=True,
        all_candidate_edges_included=True, fastmath=False, cache=False,
        classification="fixed_artificial_signature_setup_not_scientific_validation_or_benchmark",
        compilation_excluded_from_online_clocks=True, setup_in_parent_budget=True,
        signatures=signature_inventory())
    return dict(_SETUP, already_compiled=False)

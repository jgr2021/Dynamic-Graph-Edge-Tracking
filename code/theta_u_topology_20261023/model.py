"""Synthetic, fully observed interaction model for changing graph topology.

The graph controls a conditional mean, not an assumed signal precision:
    s_next = (a I - eta L(theta_t)) s + q + epsilon.
All node states s are observed synchronously without extra measurement noise;
q is a known controlled input, and epsilon is unobserved process innovation.
Using noisy node measurements in the same regression would create an
errors-in-variables problem and is NOT validated by these routines.

Positions, link restrictions, true weights and true event times belong only to
the synthetic generator/evaluator. Estimators receive H, r and sigma, never
these truth fields. This is not a model validated on real UAV measurements.
Graph seeds and observation seeds are separate explicit inputs. Formal
development/confirmation seed ranges are assigned by the parent's protocol;
the short checks below use diagnostic-only seeds within 913101--913124.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def complete_incidence(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return incidence B and lexicographic undirected edge pairs (i < j)."""
    if not isinstance(n, (int, np.integer)) or n < 4 or n % 2:
        raise ValueError("An even integer n >= 4 is required.")
    edges = np.asarray([(i, j) for i in range(n) for j in range(i + 1, n)],
                       dtype=np.int64)
    b = np.zeros((n, len(edges)), dtype=float)
    b[edges[:, 0], np.arange(len(edges))] = 1.0
    b[edges[:, 1], np.arange(len(edges))] = -1.0
    return b, edges


def laplacian(b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return (np.asarray(b) * np.asarray(theta)[None, :]) @ np.asarray(b).T


def component_count(n: int, edges: np.ndarray, theta: np.ndarray) -> int:
    """Exact positive-support connectivity; no event threshold is supplied."""
    roots = np.arange(n)

    def find(i: int) -> int:
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = int(roots[i])
        return i

    for (i, j), value in zip(edges, theta):
        if value > 0.0:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                roots[rj] = ri
    return len({find(i) for i in range(n)})


def make_graph(n: int, graph_kind: str, seed: int,
               T: int = 72) -> dict:
    """Generate exact edge removal/restoration with all-pairs candidates.

    ``bridge_failure`` has two internally complete geometric clusters and one
    permitted cross-cluster link. Its communication availability is zero at
    t=24,...,47 and restored at t=48. Other cross-cluster links are physically
    disallowed in this synthetic scenario, but are still estimator candidates.

    ``mobility_radius`` moves two rigid clusters apart and back. Edge weight is
    exp(-distance^2/(2*0.65^2)) inside communication radius 1.05 and exactly zero
    outside. Thus motion changes both positive weights and the true support.
    The schedule is defined for T >= 72; longer trajectories remain restored.
    """
    if not isinstance(T, (int, np.integer)) or T < 72:
        raise ValueError("The frozen motion schedule requires T >= 72.")
    if graph_kind not in ("bridge_failure", "mobility_radius"):
        raise ValueError(f"Unknown graph_kind: {graph_kind}")
    b, edges = complete_incidence(n)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    radii = 0.20 * np.sqrt(rng.uniform(0.0, 1.0, n))
    offsets = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    positions = np.broadcast_to(offsets, (T, n, 2)).copy()
    half = n // 2
    if graph_kind == "bridge_failure":
        separation = np.full(T, 0.70)
    else:
        separation = np.full(T, 0.65)
        separation[16:32] = np.linspace(0.65, 1.80, 16)
        separation[32:48] = 1.80
        separation[48:64] = np.linspace(1.80, 0.65, 16)
    positions[:, :half, 0] -= separation[:, None] / 2.0
    positions[:, half:, 0] += separation[:, None] / 2.0
    distances = np.linalg.norm(positions[:, edges[:, 0]]
                               - positions[:, edges[:, 1]], axis=2)
    theta = np.exp(-(distances ** 2) / (2.0 * 0.65 ** 2))
    bridge_edge = None
    if graph_kind == "mobility_radius":
        theta *= distances <= 1.05
    else:
        cross = (edges[:, 0] < half) & (edges[:, 1] >= half)
        cross_indices = np.flatnonzero(cross)
        bridge_edge = int(cross_indices[np.argmin(distances[0, cross_indices])])
        theta[:, cross] = 0.0
        theta[:, bridge_edge] = np.exp(-(distances[:, bridge_edge] ** 2)
                                      / (2.0 * 0.65 ** 2))
        theta[24:48, bridge_edge] = 0.0
    components = np.asarray([component_count(n, edges, row) for row in theta],
                            dtype=np.int64)
    support_change = np.flatnonzero(np.any((theta[1:] > 0.0)
                                           != (theta[:-1] > 0.0), axis=1)) + 1
    true_events = {
        "support_change_steps": support_change.tolist(),
        "disconnection_steps": (np.flatnonzero(np.diff(components) > 0) + 1).tolist(),
        "reconnection_steps": (np.flatnonzero(np.diff(components) < 0) + 1).tolist(),
    }
    if graph_kind == "bridge_failure":
        true_events.update(bridge_off_steps=[24], bridge_on_steps=[48])
    if not (components[0] == 1 and components[40] == 2 and components[71] == 1):
        raise ArithmeticError("Generator failed its connected/disconnected/restored contract.")
    return {
        "B": b, "edges": edges, "positions": positions, "theta": theta,
        "true_event_times": true_events, "graph_kind": graph_kind,
        "graph_seed": int(seed), "components": components,
        "bridge_edge_index": bridge_edge,
        "weight_length_scale": 0.65, "communication_radius": 1.05,
        "candidate_support": "all undirected node pairs",
    }


def generate_observations(graph: dict, seed: int, M: int = 48,
                          a: float = 0.95, sigma: float = 0.03,
                          excitation: float = 0.3) -> dict:
    """Generate observed states and the exact causal linear observation design.

    The graph is fixed for M consecutive interactions per slow step. Initial
    state is burned in for 64 transitions on the initial graph; burned states
    are not returned. The returned final state of block t is exactly the
    initial state of block t+1. A time's M*n rows are ordered by interaction,
    then node, so r[t] = H[t] @ theta[t] - epsilon[t].ravel().

    With 0 <= theta <= 1 and eta=.8/n, Laplacian eigenvalues are in [0,n].
    Transition eigenvalues are in [a-.8,a], hence [.15,.95] by default.
    This remains stable after a graph splits into disconnected components.
    """
    b = np.asarray(graph["B"], dtype=float)
    theta = np.asarray(graph["theta"], dtype=float)
    if b.ndim != 2 or theta.ndim != 2 or theta.shape[1] != b.shape[1]:
        raise ValueError("Incompatible incidence and theta dimensions.")
    if np.any(~np.isfinite(theta)) or np.any(theta < 0.0) or np.any(theta > 1.0):
        raise ValueError("The stable generator requires finite weights in [0,1].")
    if M < 1 or int(M) != M or not 0.0 < a < 1.0 or sigma <= 0.0 or excitation < 0.0:
        raise ValueError("Require M>=1, 0<a<1, sigma>0, and excitation>=0.")
    M = int(M)
    T, m = theta.shape
    n = b.shape[0]
    eta = 0.8 / n
    rng = np.random.default_rng(seed)
    state = np.zeros(n)
    f0 = a * np.eye(n) - eta * laplacian(b, theta[0])
    for _ in range(64):
        state = f0 @ state + rng.normal(0.0, excitation, n) + rng.normal(0.0, sigma, n)
    states = np.empty((T, M + 1, n))
    inputs = np.empty((T, M, n))
    noise = np.empty((T, M, n))
    h = np.empty((T, M * n, m))
    r = np.empty((T, M * n))
    for t in range(T):
        f = a * np.eye(n) - eta * laplacian(b, theta[t])
        states[t, 0] = state
        for k in range(M):
            q = rng.normal(0.0, excitation, n)
            eps = rng.normal(0.0, sigma, n)
            next_state = f @ state + q + eps
            rows = slice(k * n, (k + 1) * n)
            h[t, rows] = eta * b * (b.T @ state)[None, :]
            r[t, rows] = a * state + q - next_state
            inputs[t, k], noise[t, k] = q, eps
            states[t, k + 1] = next_state
            state = next_state
    return {
        "H": h, "r": r, "states": states, "inputs": inputs,
        "process_noise": noise, "a": float(a), "eta": float(eta),
        "sigma": float(sigma), "excitation": float(excitation),
        "theta_true": theta.copy(), "observation_seed": int(seed),
        "measurement_noise_sigma": 0.0, "burn_in_transitions": 64,
        "state_observation": "complete synchronous exact state; known input q",
    }


def run_checks(output: str | Path | None = None) -> dict:
    """Finite diagnostic-only math checks, not a scientific method batch."""
    checks = []
    for n in (6, 12):
        for kind_index, kind in enumerate(("bridge_failure", "mobility_radius")):
            graph = make_graph(n, kind, seed=913100 + n + kind_index)
            b, theta = graph["B"], graph["theta"]
            asymmetry = row_sum = 0.0
            min_lap_eig, max_lap_eig = float("inf"), -float("inf")
            min_f_eig, max_f_eig = float("inf"), -float("inf")
            for weights in theta:
                lap = laplacian(b, weights)
                eig = np.linalg.eigvalsh(lap)
                asymmetry = max(asymmetry, float(np.max(np.abs(lap - lap.T))))
                row_sum = max(row_sum, float(np.max(np.abs(lap.sum(axis=1)))))
                min_lap_eig, max_lap_eig = min(min_lap_eig, float(eig[0])), max(max_lap_eig, float(eig[-1]))
                f_eig = 0.95 - (0.8 / n) * eig
                min_f_eig, max_f_eig = min(min_f_eig, float(f_eig.min())), max(max_f_eig, float(f_eig.max()))
            short_graph = dict(graph, theta=theta[:2])
            observed = generate_observations(short_graph, seed=913108 + kind_index + n,
                                             M=max(16, n))
            identity_error = max(float(np.max(np.abs(observed["r"][t]
                                - observed["H"][t] @ theta[t]
                                + observed["process_noise"][t].ravel()))) for t in range(2))
            state = observed["states"][0, 0]
            eta = observed["eta"]
            direct_error = float(np.max(np.abs(observed["H"][0, :n] @ theta[0]
                                    - eta * laplacian(b, theta[0]) @ state)))
            individual_error = max(float(np.max(np.abs(observed["H"][0, :n, e]
                                      - eta * b[:, e] * (b[:, e] @ state))))
                                   for e in range(b.shape[1]))
            rank = int(np.linalg.matrix_rank(observed["H"][0]))
            continuity = float(np.max(np.abs(observed["states"][0, -1]
                                             - observed["states"][1, 0])))
            consensus_h = eta * b * (b.T @ np.ones(n))[None, :]
            consensus_error = float(np.max(np.abs(consensus_h)))
            observed_replay = generate_observations(short_graph, seed=913108 + kind_index + n,
                                                    M=max(16, n))
            deterministic = bool(np.array_equal(observed["H"], observed_replay["H"]))
            passed = bool(asymmetry < 1e-12 and row_sum < 1e-12 and min_lap_eig > -1e-12
                          and max_lap_eig <= n + 1e-12 and min_f_eig >= 0.15 - 1e-12
                          and max_f_eig <= 0.95 + 1e-12 and identity_error < 1e-12
                          and direct_error < 1e-12 and individual_error < 1e-12
                          and rank == b.shape[1] and continuity == 0.0
                          and consensus_error == 0.0 and deterministic)
            checks.append({
                "n": n, "graph_kind": kind, "passed": passed,
                "max_asymmetry": asymmetry, "max_row_sum": row_sum,
                "laplacian_eigenvalue_range": [min_lap_eig, max_lap_eig],
                "transition_eigenvalue_range": [min_f_eig, max_f_eig],
                "components_initial_disconnected_restored": graph["components"][[0, 40, 71]].tolist(),
                "true_event_times": graph["true_event_times"],
                "conditional_noise_identity_max_abs_error": identity_error,
                "design_direct_laplacian_max_abs_error": direct_error,
                "individual_edge_design_max_abs_error": individual_error,
                "design_rank": rank, "candidate_edge_count": b.shape[1],
                "cross_block_continuity_max_abs_error": continuity,
                "consensus_design_max_abs": consensus_error,
                "deterministic_replay": deterministic,
            })
    result = {
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "scope": "finite model diagnostics only; not a formal experiment batch",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostic_seed_scope": "913101--913124; no formal development or confirmation seeds consumed",
        "measurement_model": "exact observed s; known q; unobserved Gaussian process innovation",
        "formal_seed_policy": "development and confirmation seeds assigned by parent finite protocol",
        "checks": checks,
    }
    if output is not None:
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise AssertionError(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    receipt = run_checks(Path(__file__).with_name("CHECK_MODEL.json"))
    print(json.dumps({"status": receipt["status"], "cases": len(receipt["checks"]),
                      "source_sha256": receipt["source_sha256"]}, ensure_ascii=False))

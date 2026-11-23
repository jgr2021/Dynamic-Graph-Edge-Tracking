"""Common observation statistics for H_k = eta B diag(B.T s_k).

This standalone algebra utility does not modify or replace frozen V2 methods.
It consumes observed node states and responses only.  Every candidate and
baseline must receive the same BlockStatistics object in any future timing
comparison; its preprocessing cost must be charged once to each pipeline.

For oriented edge e=(i,j), f=(u,v), and C=sum_k s_k s_k.T,
  G[e,f] = eta**2/sigma**2 * (b_e.T b_f)
             * (C[i,u]-C[i,v]-C[j,u]+C[j,v]).
For D=sum_k s_k r_k.T,
  h[e] = eta/sigma**2 * (D[i,i]-D[i,j]-D[j,i]+D[j,j]).
These are exactly sum H_k.T H_k/sigma**2 and sum H_k.T r_k/sigma**2.
Disjoint edges have an exactly zero Gram entry.  Accumulation below visits
incident-edge blocks rather than all edge pairs; diagonal contributions are
correctly counted at both endpoints.  Returned G is deliberately dense for
compatibility, so allocation/initialization and storage still cost O(m**2).

With M observed transitions, endpoint degrees d_v, and dense node products,
preprocessing arithmetic is O(M*n**2 + sum_v d_v**2 + m**2), storage is
O(m**2 + n**2 + M*n), excluding input storage.  The m**2 term is dense output
initialization; it is not hidden by the endpoint arithmetic.  No M*n*m design
is formed.  For complete candidates sum d_v**2 = n*(n-1)**2, but m**2=O(n**4).
The existing dense solves/eigendecompositions remain O(m**3)=O(n**6).

Node common modes are removed before C/D to limit cancellation; this changes
neither H nor h because every incidence column sums to zero.  Stable residual
evaluation retains the observed M-by-n arrays and applies the Laplacian.  A
statistics-only quadratic residual is available, with its cancellation caveat.
These identities imply no exact posterior or statistical gate calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _remove_node_common_mode(values: np.ndarray) -> np.ndarray:
    # Subtract one observed node first: exact consensus then becomes exactly 0,
    # even when summing many copies of a large common value would round.
    shifted = values - values[:, :1]
    return shifted - shifted.mean(axis=1, keepdims=True)


class EdgeObservationStructure:
    """Reusable, truth-independent endpoint indexing for a simple graph.

    Edges can have either orientation; a reversed incidence column leaves its
    regression column unchanged since both B and B.T s change sign.  Candidate
    edges need not form a connected graph.  No support/weight truth is accepted.
    """

    def __init__(self, n: int, edges: np.ndarray):
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer >= 2")
        raw = np.asarray(edges)
        if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0:
            raise ValueError("Expected nonempty candidate edges[m,2]")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Candidate endpoints must have integer dtype")
        self.n = int(n)
        self.edges = raw.astype(np.int64, copy=True)
        if np.any(self.edges < 0) or np.any(self.edges >= self.n):
            raise ValueError("Candidate endpoint outside [0,n)")
        if np.any(self.edges[:, 0] == self.edges[:, 1]):
            raise ValueError("Self edges are not permitted")
        canonical = np.sort(self.edges, axis=1)
        if np.unique(canonical, axis=0).shape[0] != len(self.edges):
            raise ValueError("Duplicate undirected candidates are not permitted")
        self.m = len(self.edges)
        self.edges.setflags(write=False)
        self.incident = []
        for vertex in range(self.n):
            ids = np.flatnonzero(np.any(self.edges == vertex, axis=1))
            signs = np.where(self.edges[ids, 0] == vertex, 1.0, -1.0)
            self.incident.append((ids, signs))

    def laplacian(self, theta: np.ndarray) -> np.ndarray:
        weights = np.asarray(theta, dtype=float)
        if weights.shape != (self.m,) or not np.all(np.isfinite(weights)):
            raise ValueError("Expected finite theta[m]")
        i, j = self.edges.T
        lap = np.zeros((self.n, self.n))
        np.add.at(lap, (i, i), weights)
        np.add.at(lap, (j, j), weights)
        lap[i, j] = -weights
        lap[j, i] = -weights
        return lap

    def statistics(self, states: np.ndarray, responses: np.ndarray,
                   eta: float, sigma: float) -> BlockStatistics:
        states = np.asarray(states, dtype=float)
        responses = np.asarray(responses, dtype=float)
        eta, sigma = float(eta), float(sigma)
        if states.ndim != 2 or states.shape[1] != self.n or not len(states):
            raise ValueError("Expected nonempty observed states[M,n]")
        if responses.shape != states.shape:
            raise ValueError("Expected observed responses[M,n]")
        if not np.isfinite(eta) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Finite eta and positive finite sigma required")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(responses)):
            raise ValueError("Finite observed states and responses required")
        centered_s = _remove_node_common_mode(states)
        centered_r = _remove_node_common_mode(responses)
        C = centered_s.T @ centered_s
        D = centered_s.T @ centered_r
        G = np.zeros((self.m, self.m))
        for ids, signs in self.incident:
            if not len(ids):
                continue
            i, j = self.edges[ids].T
            edge_cov = (C[np.ix_(i, i)] - C[np.ix_(i, j)]
                        - C[np.ix_(j, i)] + C[np.ix_(j, j)])
            G[np.ix_(ids, ids)] += signs[:, None] * signs[None, :] * edge_cov
        G *= eta**2 / sigma**2
        G = (G + G.T) * 0.5
        i, j = self.edges.T
        h = (eta / sigma**2) * (D[i, i] - D[i, j] - D[j, i] + D[j, j])
        response_norm = float(np.sum(responses**2))
        if not (np.all(np.isfinite(G)) and np.all(np.isfinite(h))
                and np.isfinite(response_norm)):
            raise FloatingPointError("Nonfinite accumulated observation statistics")
        return BlockStatistics(self, G, h, response_norm / sigma**2,
                               centered_s, responses.copy(), eta, sigma)


@dataclass(frozen=True)
class BlockStatistics:
    structure: EdgeObservationStructure
    G: np.ndarray
    h: np.ndarray
    response_norm_squared_scaled: float
    centered_states: np.ndarray
    responses: np.ndarray
    eta: float
    sigma: float

    @property
    def row_count(self) -> int:
        return int(self.responses.size)

    def prediction(self, theta: np.ndarray) -> np.ndarray:
        """Return H theta as M-by-n responses, without constructing H.

        Dense node multiplication costs O(M*n**2); Laplacian construction costs
        O(n**2+m).  This is not claimed to be optimal for sparse candidates.
        """
        return self.eta * (self.centered_states @ self.structure.laplacian(theta))

    def residual(self, theta: np.ndarray) -> np.ndarray:
        return self.responses - self.prediction(theta)

    def residual_score(self, theta: np.ndarray) -> np.ndarray:
        """Return H.T(r-H theta)/sigma**2 using the common G/h."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.h.shape or not np.all(np.isfinite(theta)):
            raise ValueError("Expected finite theta[m]")
        return self.h - self.G @ theta

    def residual_squared(self, theta: np.ndarray, *, stable: bool = True) -> float:
        if stable:
            innovation = self.residual(theta)
            return float(np.sum(innovation**2))
        # Algebraically exact but subtracts large terms near an exact fit.
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.h.shape or not np.all(np.isfinite(theta)):
            raise ValueError("Expected finite theta[m]")
        return float(self.sigma**2 * (self.response_norm_squared_scaled
                                     - 2.0 * self.h @ theta
                                     + theta @ self.G @ theta))

    def predictive_nis(self, theta: np.ndarray, J_previous: np.ndarray) -> float:
        """Frozen V2 variational NIS identity with no repeated Gram or H.

        One m-by-m Cholesky remains O(m**3).  The stable residual path uses
        observed M-by-n arrays.  This is an algebraic equivalence, not evidence
        that J_previous is calibrated or that its gate has a chi-square law.
        """
        J_previous = np.asarray(J_previous, dtype=float)
        if J_previous.shape != self.G.shape or not np.all(np.isfinite(J_previous)):
            raise ValueError("Expected finite J_previous[m,m]")
        A = J_previous + self.G
        A = (A + A.T) * 0.5
        correction = cho_solve(cho_factor(A, lower=True), self.residual_score(theta))
        remaining = self.residual(np.asarray(theta) + correction)
        value = float(np.sum(remaining**2) / self.sigma**2
                      + correction @ J_previous @ correction)
        if not np.isfinite(value) or value < -1e-8:
            raise FloatingPointError("Invalid predictive NIS")
        return max(0.0, value)

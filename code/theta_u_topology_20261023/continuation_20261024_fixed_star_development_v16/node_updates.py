"""Observation-only node-statistic sparse-increment point estimators.

No function in a node method forms G[m,m], carries P, solves a linear system,
or calls an eigensolver.  Dense conversion and matched dense optimizers are
explicit separate interfaces for baseline pipelines and algebra comparisons.

Every method minimizes the CURRENT block's convex objective on [0,1]^m:
  .5 ||r-H theta||^2/sigma^2 + alpha.T theta
  + .5 rho ||theta-anchor||^2 + beta.T abs(theta-anchor),
where anchor is its own previous estimate, rho=1e-8, theta_0=.3, and
alpha=a*sqrt(2 log(max(2,m)) diag(G)), beta=b*the same scale.
Node/dense pairs have identical objectives and precision.  At the FIRST block
beta is zero for all methods.  All candidate coordinates are visited, including
zero edges; no true support, event time, graph position, or physical gap enters.

With C=sum s s.T and Z=L(theta) C, changing edge e by delta updates only rows
i,j of Z: Z += delta b_e (b_e.T C). Each coordinate costs O(n), a full sweep
O(m*n). At each sweep boundary L and Z are independently rebuilt, drift is
checked, and ALL coordinate KKT conditions are checked with the refreshed
cache. Thus actual time includes O(S*(n^3+m*n+m)) in the current implementation,
not just O(number_of_changed_edges*n). Statistics cost O(M*n^2+m), storage
O(n^2+m+M*n). Full current node covariance plus rho is mathematically PSD/PD;
no covariance/posterior calibration claim is made.

Run this module with --self-test only for deterministic algebra checks.  It
never automatically starts any development, confirmation, or timing campaign.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

from structured_stats import BlockStatistics, EdgeObservationStructure


MAX_SWEEPS = 5000
RELATIVE_KKT_TOLERANCE = 1e-8
RHO = 1e-8
DRIFT_ROUNDOFF_FACTOR = 128.0
METHOD_SPECS = {
    "node_block_l1": (0.5, 0.0),
    # Before any scientific data generation, (.1,.5) was rejected by the
    # noiseless scalar deletion derivative alpha-beta-rho*anchor < 0.
    # These fixed (.2,.1) coefficients reflect that mathematical correction,
    # not a selection based on development/confirmation results.
    "node_fused_l1": (0.2, 0.1),
    "node_fused_beta0": (0.2, 0.0),
}
_SETUP = None


def _center(values):
    shifted = values-values[:, :1]
    return shifted-shifted.mean(axis=1, keepdims=True)


@dataclass(frozen=True)
class NodeStatistics:
    structure: EdgeObservationStructure
    C: np.ndarray
    D: np.ndarray
    h: np.ndarray
    Gdiag: np.ndarray
    states: np.ndarray
    centered_states: np.ndarray
    responses: np.ndarray
    eta: float
    sigma: float
    response_norm_squared_scaled: float

    @classmethod
    def from_observations(cls, structure, states, responses, eta, sigma):
        states, responses = np.asarray(states, dtype=float), np.asarray(responses, dtype=float)
        eta, sigma = float(eta), float(sigma)
        if not isinstance(structure, EdgeObservationStructure):
            raise TypeError("An EdgeObservationStructure is required")
        if states.ndim != 2 or states.shape[1] != structure.n or not len(states):
            raise ValueError("Expected nonempty observed states[M,n]")
        if responses.shape != states.shape:
            raise ValueError("Expected observed responses[M,n]")
        if not np.isfinite(eta) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Finite eta and positive finite sigma required")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(responses)):
            raise ValueError("Finite observed arrays required")
        centered_s, centered_r = _center(states), _center(responses)
        C, D = centered_s.T@centered_s, centered_s.T@centered_r
        C = (C+C.T)*0.5
        i, j = structure.edges.T
        h = eta/sigma**2 * (D[i,i]-D[i,j]-D[j,i]+D[j,j])
        Gdiag = 2.0*eta**2/sigma**2 * (C[i,i]-C[i,j]-C[j,i]+C[j,j])
        # A genuinely negative squared-design diagonal is never hidden by a
        # ridge or a changed tolerance. Exact-consensus/identical-node columns
        # yield exactly zero after the common-mode subtraction.
        if np.any(Gdiag < 0) or not np.all(np.isfinite(Gdiag)):
            raise FloatingPointError("Negative/nonfinite observation curvature")
        response_norm = float(np.sum(responses**2)/sigma**2)
        if not (np.all(np.isfinite(C)) and np.all(np.isfinite(D))
                and np.all(np.isfinite(h)) and np.isfinite(response_norm)):
            raise FloatingPointError("Nonfinite node statistics")
        return cls(structure, np.ascontiguousarray(C), np.ascontiguousarray(D),
                   np.ascontiguousarray(h), np.ascontiguousarray(Gdiag),
                   states.copy(), np.ascontiguousarray(centered_s), responses.copy(),
                   eta, sigma, response_norm)

    @property
    def row_count(self):
        return int(self.responses.size)

    def prediction(self, theta):
        return self.eta*(self.centered_states@self.structure.laplacian(theta))

    def residual(self, theta):
        return self.responses-self.prediction(theta)

    def residual_squared(self, theta, *, stable=True):
        if stable:
            return float(np.sum(self.residual(theta)**2))
        theta = _theta_vector(theta, self.structure.m)
        L = self.structure.laplacian(theta)
        data_quad = self.eta**2/self.sigma**2 * float(np.sum((L@self.C)*L))
        return float(self.sigma**2*(self.response_norm_squared_scaled-2*self.h@theta+data_quad))

    def residual_score(self, theta):
        """H.T(r-H theta)/sigma^2 from the node operator, without dense G."""
        theta = _theta_vector(theta, self.structure.m)
        Z = self.structure.laplacian(theta)@self.C
        i, j = self.structure.edges.T
        return self.h-self.eta**2/self.sigma**2*(Z[i,i]-Z[i,j]-Z[j,i]+Z[j,j])

    def observation_residual_score(self, theta):
        """Independent final score from actual observed-state residuals.

        This extra O(M*n^2+m) check prevents a drifting/small-cancellation
        node cache from falsely passing the declared KKT threshold.
        """
        return _observed_score(self,theta)

    def to_block_statistics(self):
        """Expand THIS C/D to legacy dense G; time/charge conversion separately.

        No states are revisited and no H is constructed. Output allocation is
        O(m^2), incident blocks cost O(sum_v d_v^2), and dense storage O(m^2).
        Node estimators never call this function.
        """
        G = np.zeros((self.structure.m, self.structure.m))
        for ids, signs in self.structure.incident:
            if not len(ids):
                continue
            i, j = self.structure.edges[ids].T
            covariance = (self.C[np.ix_(i,i)]-self.C[np.ix_(i,j)]
                          -self.C[np.ix_(j,i)]+self.C[np.ix_(j,j)])
            G[np.ix_(ids,ids)] += signs[:,None]*signs[None,:]*covariance
        G *= self.eta**2/self.sigma**2
        G = (G+G.T)*0.5
        return BlockStatistics(self.structure, G, self.h.copy(), self.response_norm_squared_scaled,
                               self.centered_states, self.responses, self.eta, self.sigma)


def _theta_vector(theta, m):
    result = np.asarray(theta, dtype=float)
    if result.shape != (m,) or not np.all(np.isfinite(result)):
        raise ValueError("Expected finite theta[m]")
    return result


def _observed_score(stats,theta):
    residual = stats.residual(theta)
    cross = stats.centered_states.T@_center(residual)
    i,j = stats.structure.edges.T
    return stats.eta/stats.sigma**2*(cross[i,i]-cross[i,j]-cross[j,i]+cross[j,j])


@njit(cache=False)
def _laplacian(n, edges, theta):
    L = np.zeros((n,n))
    for e in range(len(theta)):
        i, j = edges[e,0], edges[e,1]
        L[i,i] += theta[e]; L[j,j] += theta[e]
        L[i,j] -= theta[e]; L[j,i] -= theta[e]
    return L


@njit(cache=False)
def _node_gradient(Z, h, theta, anchor, alpha, edges, scale, rho):
    gradient = np.empty(len(theta))
    for e in range(len(theta)):
        i, j = edges[e,0], edges[e,1]
        gradient[e] = (scale*(Z[i,i]-Z[i,j]-Z[j,i]+Z[j,j])-h[e]
                       +rho*(theta[e]-anchor[e])+alpha[e])
    return gradient


@njit(cache=False)
def _kkt(gradient, theta, anchor, beta):
    """Distance to the exact interval subgradient + box normal cone.

    No near-anchor or near-boundary tolerance reclassifies a coordinate.
    Soft thresholding generates exact anchor equality and box endpoints.
    """
    residual = 0.0
    for e in range(len(theta)):
        if theta[e] > anchor[e]:
            lower = upper = gradient[e]+beta[e]
        elif theta[e] < anchor[e]:
            lower = upper = gradient[e]-beta[e]
        else:
            lower, upper = gradient[e]-beta[e], gradient[e]+beta[e]
        if theta[e] <= 0.0:
            violation = max(-upper,0.0)
        elif theta[e] >= 1.0:
            violation = max(lower,0.0)
        else:
            violation = max(lower, -upper, 0.0)
        residual = max(residual, violation)
    return residual


@njit(cache=False)
def _prox_coordinate(theta, anchor, gradient, curvature, beta):
    value = theta-anchor-gradient/curvature
    threshold = beta/curvature
    if value > threshold:
        changed = value-threshold
    elif value < -threshold:
        changed = value+threshold
    else:
        changed = 0.0
    return min(max(anchor+changed,0.0),1.0)


@njit(cache=False)
def _node_cd(C,h,Gdiag,edges,scale,anchor,alpha,beta,rho,initial,max_sweeps,relative_tol):
    theta = np.minimum(np.maximum(initial.copy(),0.0),1.0)
    n, m = C.shape[0], len(theta)
    L = _laplacian(n,edges,theta)
    Z = L@C
    gradient = _node_gradient(Z,h,theta,anchor,alpha,edges,scale,rho)
    normalizer = max(1.0,np.max(np.abs(h+rho*anchor-alpha)))
    tolerance = relative_tol*normalizer
    residual = _kkt(gradient,theta,anchor,beta)
    sweeps = visits = updates = 0
    drift_L = drift_Z = guard_L_max = guard_Z_max = 0.0
    refreshes = 1
    while residual > tolerance and sweeps < max_sweeps:
        updates_this_sweep = 0
        for e in range(m):
            visits += 1
            i, j = edges[e,0], edges[e,1]
            g = (scale*(Z[i,i]-Z[i,j]-Z[j,i]+Z[j,j])-h[e]
                 +rho*(theta[e]-anchor[e])+alpha[e])
            new = _prox_coordinate(theta[e],anchor[e],g,Gdiag[e]+rho,beta[e])
            delta = new-theta[e]
            if delta != 0.0:
                theta[e] = new
                L[i,i] += delta; L[j,j] += delta
                L[i,j] -= delta; L[j,i] -= delta
                for k in range(n):
                    value = delta*(C[i,k]-C[j,k])
                    Z[i,k] += value; Z[j,k] -= value
                updates += 1
                updates_this_sweep += 1
        sweeps += 1
        # Rebuild both caches, so neither an accumulated diagonal degree sum
        # nor an incremented Z can certify itself. Refresh cost is paid.
        actual_L = _laplacian(n,edges,theta)
        actual_Z = actual_L@C
        error_L, error_Z = np.max(np.abs(actual_L-L)), np.max(np.abs(actual_Z-Z))
        amplitude_L, amplitude_Z = max(1.0,np.max(np.abs(actual_L))), max(1.0,np.max(np.abs(actual_Z)))
        roundoff = DRIFT_ROUNDOFF_FACTOR*np.finfo(np.float64).eps*(1+updates_this_sweep)*max(1,n)
        guard_L, guard_Z = roundoff*amplitude_L, roundoff*amplitude_Z
        if not np.isfinite(error_L+error_Z) or error_L > guard_L or error_Z > guard_Z:
            raise FloatingPointError("Node cache drift exceeds its roundoff guard")
        drift_L, drift_Z = max(drift_L,error_L), max(drift_Z,error_Z)
        guard_L_max, guard_Z_max = max(guard_L_max,guard_L), max(guard_Z_max,guard_Z)
        L, Z = actual_L, actual_Z
        refreshes += 1
        gradient = _node_gradient(Z,h,theta,anchor,alpha,edges,scale,rho)
        residual = _kkt(gradient,theta,anchor,beta)
    return (theta,L,Z,gradient,sweeps,visits,updates,residual,tolerance,normalizer,
            refreshes,drift_L,drift_Z,guard_L_max,guard_Z_max)


@njit(cache=False)
def _dense_cd(G,h,anchor,alpha,beta,rho,initial,max_sweeps,relative_tol):
    theta = np.minimum(np.maximum(initial.copy(),0.0),1.0)
    gradient = G@theta-h+rho*(theta-anchor)+alpha
    normalizer = max(1.0,np.max(np.abs(h+rho*anchor-alpha)))
    tolerance = relative_tol*normalizer
    residual = _kkt(gradient,theta,anchor,beta)
    sweeps = visits = updates = 0
    while residual > tolerance and sweeps < max_sweeps:
        for e in range(len(theta)):
            visits += 1
            new = _prox_coordinate(theta[e],anchor[e],gradient[e],G[e,e]+rho,beta[e])
            delta = new-theta[e]
            if delta != 0.0:
                theta[e] = new
                for j in range(len(theta)):
                    gradient[j] += delta*G[j,e]
                gradient[e] += delta*rho
                updates += 1
        sweeps += 1
        gradient = G@theta-h+rho*(theta-anchor)+alpha
        residual = _kkt(gradient,theta,anchor,beta)
    return theta,gradient,sweeps,visits,updates,residual,tolerance,normalizer


def setup():
    """Compile both node and matched dense kernels once, outside online time."""
    global _SETUP
    if _SETUP is not None:
        return {**_SETUP,"already_compiled":True}
    structure = EdgeObservationStructure(3,np.array([[0,1],[0,2],[1,2]],dtype=np.int64))
    C, h, diagonal = np.eye(3), np.ones(3), np.full(3,4.0)
    anchor, alpha, beta = np.full(3,.3), np.full(3,.1), np.full(3,.2)
    started = time.perf_counter()
    _node_cd(C,h,diagonal,structure.edges,1.0,anchor,alpha,beta,RHO,anchor,MAX_SWEEPS,RELATIVE_KKT_TOLERANCE)
    node_s = time.perf_counter()-started
    started = time.perf_counter()
    _dense_cd(np.eye(3)*4,h,anchor,alpha,beta,RHO,anchor,MAX_SWEEPS,RELATIVE_KKT_TOLERANCE)
    dense_s = time.perf_counter()-started
    _SETUP = dict(node_jit_seconds=node_s,dense_jit_seconds=dense_s,fastmath=False,cache=False,
                  module_import_time_included=False,excluded_from_online_time=True)
    return {**_SETUP,"already_compiled":False}


class NodeMethod:
    def __init__(self,structure,name):
        if name not in METHOD_SPECS:
            raise ValueError("Unknown node method")
        self.structure,self.name = structure,name
        self.alpha_factor,self.beta_factor = METHOD_SPECS[name]
        self.theta,self.step = np.full(structure.m,.3),0
        self.last = {}

    def audit_state(self,copy=True):
        result = dict(theta=self.theta,step=self.step,**self.last)
        return {key:value.copy() if copy and isinstance(value,np.ndarray) else value for key,value in result.items()}

    def update(self,stats):
        if (not isinstance(stats,NodeStatistics) or stats.structure.n != self.structure.n
                or not np.array_equal(stats.structure.edges,self.structure.edges)):
            raise ValueError("Matching NodeStatistics/candidate structure required")
        anchor = self.theta.copy()
        regularizer_scale = np.sqrt(2*np.log(max(2,self.structure.m))*stats.Gdiag)
        alpha = self.alpha_factor*regularizer_scale
        beta = (0.0 if self.step == 0 else self.beta_factor)*regularizer_scale
        values = _node_cd(stats.C,stats.h,stats.Gdiag,self.structure.edges,stats.eta**2/stats.sigma**2,
                          anchor,alpha,beta,RHO,anchor,MAX_SWEEPS,RELATIVE_KKT_TOLERANCE)
        (theta,L,Z,gradient,sweeps,visits,updates,residual,tolerance,normalizer,
         refreshes,drift_L,drift_Z,guard_L,guard_Z) = values
        independent_gradient = -stats.observation_residual_score(theta)+RHO*(theta-anchor)+alpha
        if not all(np.all(np.isfinite(value)) for value in (theta,L,Z,gradient,independent_gradient)):
            raise FloatingPointError("Nonfinite node solve or observed-residual gradient")
        observed_residual = float(_kkt(independent_gradient,theta,anchor,beta))
        # No tolerance enlargement: both node-cache and observed-residual KKT
        # must satisfy exactly the same threshold before success is reported.
        success = bool(np.all(np.isfinite(theta)) and residual <= tolerance and observed_residual <= tolerance)
        failure = None if success else ("observed_residual_kkt_check_failed"
                  if residual <= tolerance else "coordinate_descent_kkt_budget_exhausted")
        self.theta,self.step = theta.copy(),self.step+1
        self.last = dict(anchor=anchor,C=stats.C.copy(),D=stats.D.copy(),h=stats.h.copy(),
                         Gdiag=stats.Gdiag.copy(),L=L,Z=Z,alpha=alpha,beta=beta,rho=RHO,
                         gradient=independent_gradient)
        meta = _metadata(self.name,stats,theta,anchor,alpha,beta,sweeps,visits,updates,
                         max(float(residual),observed_residual),float(tolerance),float(normalizer),success,failure)
        meta.update(node_cache_kkt_residual=float(residual),observed_residual_kkt_residual=observed_residual,
                    step=self.step,beta_first_block_disabled=bool(self.step==1),
                    cache_refreshes=int(refreshes),cache_drift_L_max_abs=float(drift_L),cache_drift_Z_max_abs=float(drift_Z),
                    cache_drift_L_guard=float(guard_L),cache_drift_Z_guard=float(guard_Z),cache_drift_guard_exceeded=False,
                    dense_gram_constructed=False,full_covariance_carried=False,
                    full_kkt_checks=int(sweeps+2),kkt_coordinates_checked=int((sweeps+2)*self.structure.m))
        return theta.copy(),meta


class DenseSameObjectiveMethod:
    def __init__(self,structure,node_name):
        if node_name not in METHOD_SPECS:
            raise ValueError("Unknown dense counterpart")
        self.structure,self.node_name = structure,node_name
        self.name = node_name.replace("node_","dense_",1)
        self.alpha_factor,self.beta_factor = METHOD_SPECS[node_name]
        self.theta,self.step = np.full(structure.m,.3),0
        self.last = {}

    def audit_state(self,copy=True):
        result = dict(theta=self.theta,step=self.step,**self.last)
        return {key:value.copy() if copy and isinstance(value,np.ndarray) else value for key,value in result.items()}

    def update(self,stats):
        if (not isinstance(stats,BlockStatistics) or stats.G.shape != (self.structure.m,self.structure.m)
                or stats.structure.n != self.structure.n
                or not np.array_equal(stats.structure.edges,self.structure.edges)):
            raise ValueError("Matching legacy dense BlockStatistics required")
        anchor = self.theta.copy()
        diagonal = np.diag(stats.G)
        if np.any(diagonal < 0):
            raise FloatingPointError("Negative dense observation curvature")
        regularizer_scale = np.sqrt(2*np.log(max(2,self.structure.m))*diagonal)
        alpha = self.alpha_factor*regularizer_scale
        beta = (0.0 if self.step == 0 else self.beta_factor)*regularizer_scale
        theta,gradient,sweeps,visits,updates,residual,tolerance,normalizer = _dense_cd(
            stats.G,stats.h,anchor,alpha,beta,RHO,anchor,MAX_SWEEPS,RELATIVE_KKT_TOLERANCE)
        independent_gradient = -_observed_score(stats,theta)+RHO*(theta-anchor)+alpha
        if not all(np.all(np.isfinite(value)) for value in (theta,gradient,independent_gradient)):
            raise FloatingPointError("Nonfinite dense solve or observed-residual gradient")
        observed_residual = float(_kkt(independent_gradient,theta,anchor,beta))
        success = bool(np.all(np.isfinite(theta)) and residual <= tolerance and observed_residual <= tolerance)
        self.theta,self.step = theta.copy(),self.step+1
        self.last = dict(anchor=anchor,G=stats.G.copy(),h=stats.h.copy(),Gdiag=diagonal.copy(),
                         alpha=alpha,beta=beta,rho=RHO,gradient=independent_gradient)
        failure = None if success else ("observed_residual_kkt_check_failed" if residual <= tolerance
                                        else "coordinate_descent_kkt_budget_exhausted")
        meta = _metadata(self.name,stats,theta,anchor,alpha,beta,sweeps,visits,updates,max(float(residual),observed_residual),
                         float(tolerance),float(normalizer),success,failure)
        meta.update(dense_gram_constructed=True,full_covariance_carried=False,
                    step=self.step,beta_first_block_disabled=bool(self.step==1),
                    dense_gradient_kkt_residual=float(residual),observed_residual_kkt_residual=observed_residual,
                    full_kkt_checks=int(sweeps+2),kkt_coordinates_checked=int((sweeps+2)*self.structure.m),
                    dense_conversion_must_be_charged_separately=True)
        return theta.copy(),meta


def _metadata(name,stats,theta,anchor,alpha,beta,sweeps,visits,updates,residual,tolerance,normalizer,success,failure):
    rss = stats.residual_squared(theta,stable=True)
    objective = (.5*rss/stats.sigma**2+alpha@theta+.5*RHO*np.sum((theta-anchor)**2)+beta@np.abs(theta-anchor))
    if not np.isfinite(rss) or not np.isfinite(objective):
        raise FloatingPointError("Nonfinite actual residual or convex objective")
    return dict(method=name,optimizer="node_proximal_cyclic_cd" if name.startswith("node_") else "dense_proximal_cyclic_cd",
        success=bool(success),converged=bool(success),failure_reason=failure,iterations=int(sweeps),
        coordinate_visits=int(visits),coordinate_updates=int(updates),
        actual_changed_coordinates=int(np.count_nonzero(theta-anchor)),
        actual_changed_estimated_edges=int(np.count_nonzero(theta-anchor)),
        kkt_residual=float(residual),kkt_relative_residual=float(residual/normalizer),kkt_tolerance=float(tolerance),
        max_sweeps=MAX_SWEEPS,relative_kkt_tolerance=RELATIVE_KKT_TOLERANCE,objective=float(objective),
        block_residual_squared=float(rss),block_rows=int(stats.row_count),rho=RHO,
        alpha_min=float(np.min(alpha)),alpha_max=float(np.max(alpha)),beta_min=float(np.min(beta)),beta_max=float(np.max(beta)),
        beta_penalty_active=bool(np.count_nonzero(beta)>0),
        posterior_covariance_claimed=False,triggered=False,uses_truth_support=False,uses_event_times=False,
        current_block_only=True,anchor="own_previous_point_estimate",warm_start="own_previous_point_estimate")


def make_node_methods(structure):
    return {name:NodeMethod(structure,name) for name in METHOD_SPECS}


def make_dense_methods(structure):
    result = {}
    for name in METHOD_SPECS:
        method = DenseSameObjectiveMethod(structure,name)
        result[method.name] = method
    return result


def _dense_reference(n,edges,states,responses,eta,sigma):
    B = np.zeros((n,len(edges)))
    B[edges[:,0],np.arange(len(edges))] = 1
    B[edges[:,1],np.arange(len(edges))] = -1
    H = np.vstack([eta*B*(B.T@s)[None,:] for s in states])
    r = responses.reshape(-1)
    return H,r,H.T@H/sigma**2,H.T@r/sigma**2


def self_check():
    """Deterministic algebra only; never read formal observation/results files."""
    from scipy.optimize import Bounds, LinearConstraint, minimize

    root = Path(__file__).resolve().parent
    inherited = ("model.py","rng_history.py","structured_stats.py","fast_ewls.py","matched_kf.py","spectral.py",
                 "vendor/model_v2.py","vendor/spectrum_fast.py")
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    before = {name:sha(root/name) for name in inherited}
    compile_receipt = setup()
    rng = np.random.default_rng(9194831)
    checks = []
    for n,M in ((3,4),(6,9),(12,48),(24,4),(48,3)):
        edges = np.array([(i,j) for i in range(n) for j in range(i+1,n)],dtype=np.int64)
        structure = EdgeObservationStructure(n,edges)
        states,responses = rng.normal(size=(M,n)),rng.normal(size=(M,n))
        stats = NodeStatistics.from_observations(structure,states,responses,.8/n,.03)
        dense = stats.to_block_statistics()
        H,r,G,h = _dense_reference(n,edges,states,responses,.8/n,.03)
        theta = rng.uniform(0,1,len(edges))
        assert not hasattr(stats,"G")
        assert np.allclose(dense.G,G,atol=4e-10,rtol=4e-12)
        assert np.allclose(stats.h,h,atol=4e-10,rtol=4e-12)
        assert np.allclose(stats.Gdiag,np.diag(G),atol=4e-10,rtol=4e-12)
        assert np.allclose(stats.prediction(theta).reshape(-1),H@theta,atol=2e-13,rtol=2e-13)
        assert np.allclose(stats.residual_score(theta),H.T@(r-H@theta)/.03**2,atol=2e-9,rtol=2e-11)
        checks.append(dict(case="node_statistics_dense_observation_equivalence",n=n,M=M,
                           G_max_abs_error=float(np.max(np.abs(dense.G-G))),h_max_abs_error=float(np.max(np.abs(stats.h-h)))))
    for case,n,M,blocks in (("excited_multiblock",6,12,4),("consensus",6,8,3),("large_common_modes",4,9,3)):
        edges = np.array([(i,j) for i in range(n) for j in range(i+1,n)],dtype=np.int64)
        structure = EdgeObservationStructure(n,edges)
        node,dense = make_node_methods(structure),make_dense_methods(structure)
        max_error,max_objective,max_relative_kkt,max_drift_L,max_drift_Z = 0.,0.,0.,0.,0.
        total_updates,total_changed = 0,0
        for step in range(blocks):
            states = rng.normal(size=(M,n))
            if case == "consensus":
                states = np.repeat(rng.normal(size=(M,1)),n,axis=1)
            if case == "large_common_modes":
                states += 1e6*rng.normal(size=(M,1))
            H,_,_,_ = _dense_reference(n,edges,states,np.zeros((M,n)),.8/n,.03)
            truth = np.full(len(edges),.3)
            truth[0] = .85 if step >= 2 else .2
            responses = (H@truth+rng.normal(0,.03,M*n)).reshape(M,n)
            if case == "large_common_modes":
                responses += 1e5*rng.normal(size=(M,1))
            stats = NodeStatistics.from_observations(structure,states,responses,.8/n,.03)
            block = stats.to_block_statistics()
            for name,method in node.items():
                theta,meta = method.update(stats)
                expected,reference_meta = dense[name.replace("node_","dense_",1)].update(block)
                assert meta["success"] and reference_meta["success"],(case,name,step,meta)
                error = float(np.max(np.abs(theta-expected)))
                assert error < 3e-7,(case,name,step,error)
                assert np.array_equal(method.last["beta"],dense[name.replace("node_","dense_",1)].last["beta"])
                if step == 0:
                    assert np.count_nonzero(method.last["beta"]) == 0
                assert method.last["L"].shape==(n,n) and method.last["Z"].shape==(n,n)
                assert not any(key in method.audit_state() for key in ("G","P"))
                max_error = max(max_error,error)
                max_objective = max(max_objective,abs(meta["objective"]-reference_meta["objective"]))
                max_relative_kkt = max(max_relative_kkt,meta["kkt_relative_residual"])
                max_drift_L = max(max_drift_L,meta["cache_drift_L_max_abs"])
                max_drift_Z = max(max_drift_Z,meta["cache_drift_Z_max_abs"])
                total_updates += meta["coordinate_updates"]
                total_changed += meta["actual_changed_coordinates"]
        checks.append(dict(case=case,method_updates=blocks*3,max_theta_abs_error=max_error,
                           max_objective_abs_error=max_objective,max_relative_kkt=max_relative_kkt,
                           total_coordinate_updates=total_updates,total_final_changed_coordinates=total_changed,
                           max_cache_drift_L=max_drift_L,max_cache_drift_Z=max_drift_Z,
                           all_node_dense_converged=True))
    # Independent epigraph-constrained SLSQP reference for the nonsmooth
    # objective; it shares neither the coordinate update nor its KKT routine.
    for n in (3,4):
        edges = np.array([(i,j) for i in range(n) for j in range(i+1,n)],dtype=np.int64)
        structure = EdgeObservationStructure(n,edges)
        m,M = len(edges),10
        states,responses = rng.normal(size=(M,n)),rng.normal(size=(M,n))
        stats = NodeStatistics.from_observations(structure,states,responses,.8/n,.3)
        H,r,G,h = _dense_reference(n,edges,states,responses,.8/n,.3)
        anchor = rng.uniform(0,1,m)
        alpha = .2*np.sqrt(2*np.log(max(2,m))*np.diag(G))
        beta = .1*np.sqrt(2*np.log(max(2,m))*np.diag(G))
        values = _node_cd(stats.C,stats.h,stats.Gdiag,structure.edges,stats.eta**2/stats.sigma**2,
                          anchor,alpha,beta,RHO,anchor,MAX_SWEEPS,RELATIVE_KKT_TOLERANCE)
        actual = values[0]
        J,c = G+RHO*np.eye(m),h+RHO*anchor-alpha
        objective_scale = max(1.,float(np.linalg.norm(J,np.inf)),float(np.max(np.abs(c))))
        def objective(x):
            theta,epigraph = x[:m],x[m:]
            return (.5*theta@J@theta-c@theta+beta@epigraph)/objective_scale
        def jacobian(x):
            return np.r_[J@x[:m]-c,beta]/objective_scale
        A = np.vstack((np.column_stack((-np.eye(m),np.eye(m))),np.column_stack((np.eye(m),np.eye(m)))))
        reference = minimize(objective,np.r_[anchor,np.zeros(m)],jac=jacobian,method="SLSQP",
            bounds=Bounds(np.zeros(2*m),np.r_[np.ones(m),np.full(m,np.inf)]),
            constraints=[LinearConstraint(A,np.r_[-anchor,anchor],np.full(2*m,np.inf))],
            options={"ftol":1e-13,"maxiter":10000})
        assert reference.success,reference.message
        assert values[7] <= values[8]
        error = float(np.max(np.abs(actual-reference.x[:m])))
        assert error < 3e-6,(n,error)
        actual_objective = (.5*actual@J@actual-c@actual+beta@np.abs(actual-anchor))/objective_scale
        assert abs(actual_objective-reference.fun) < 2e-10
        checks.append(dict(case="independent_epigraph_SLSQP",n=n,max_theta_abs_error=error,
                           scaled_objective_abs_error=abs(actual_objective-reference.fun)))
    # Independent finite candidate enumeration for a scalar convex piecewise
    # quadratic, including the box/kink intersections p=0 and p=1.
    scalar_count=0
    for p in (0.,.37,1.):
        for linear in (-1.,0.,.5,2.,3.):
            q,b=2.,.4
            candidates=[0.,p,1.]
            below,above=(linear+b)/q,(linear-b)/q
            if 0 <= below <= p:
                candidates.append(below)
            if p <= above <= 1:
                candidates.append(above)
            value=lambda x:.5*q*x*x-linear*x+b*abs(x-p)
            expected=min(candidates,key=value)
            actual=float(_prox_coordinate(.6,p,q*.6-linear,q,b))
            assert abs(actual-expected) < 2e-15
            residual=float(_kkt(np.array([q*actual-linear]),np.array([actual]),np.array([p]),np.array([b])))
            assert residual < 2e-14
            scalar_count+=1
    assert _kkt(np.array([-1.]),np.array([0.]),np.array([0.]),np.array([.4])) > 0
    assert _kkt(np.array([1.]),np.array([1.]),np.array([1.]),np.array([.4])) > 0
    checks.append(dict(case="box_kink_one_sided_KKT_and_exact_scalar_oracle",scalar_cases=scalar_count,passed=True))
    # Pure scalar theory guard: when r=0, theta=0 must remain attainable from
    # a positive anchor. With the pre-theory (.1,.5) coefficients, its right
    # derivative would instead be negative, so exact deletion is not optimal.
    structure=EdgeObservationStructure(2,np.array([[0,1]],dtype=np.int64))
    states=np.array([[1.,-1.],[-1.,1.],[.5,-.5],[-.5,.5]])
    zero_response=np.zeros_like(states)
    deleted_stats=NodeStatistics.from_observations(structure,states,zero_response,.4,.03)
    scale=float(np.sqrt(2*np.log(2)*deleted_stats.Gdiag[0]))
    anchor=.7
    old_right_derivative=(.1-.5)*scale-RHO*anchor
    new_right_derivative=(.2-.1)*scale-RHO*anchor
    assert old_right_derivative < 0 and new_right_derivative > 0
    node=make_node_methods(structure)["node_fused_l1"]
    dense=make_dense_methods(structure)["dense_fused_l1"]
    node.theta[:]=anchor;dense.theta[:]=anchor;node.step=dense.step=1
    deleted,delete_meta=node.update(deleted_stats)
    dense_deleted,dense_delete_meta=dense.update(deleted_stats.to_block_statistics())
    assert delete_meta["success"] and dense_delete_meta["success"]
    assert deleted[0] == dense_deleted[0] == 0.
    restore_response=deleted_stats.prediction(np.array([.8]))
    restored_stats=NodeStatistics.from_observations(structure,states,restore_response,.4,.03)
    restored,restore_meta=node.update(restored_stats)
    dense_restored,dense_restore_meta=dense.update(restored_stats.to_block_statistics())
    assert restore_meta["success"] and dense_restore_meta["success"]
    assert restored[0] > 0 and np.allclose(restored,dense_restored,rtol=0,atol=1e-13)
    expected_restore=(restored_stats.h[0]-.3*scale)/(restored_stats.Gdiag[0]+RHO)
    assert np.isclose(restored[0],expected_restore,atol=1e-13,rtol=1e-13)
    checks.append(dict(case="noiseless_scalar_deletion_and_restoration_theory_guard",passed=True,
        pretheory_alpha_factor=.1,pretheory_beta_factor=.5,
        pretheory_deletion_right_derivative=old_right_derivative,
        fixed_alpha_factor=.2,fixed_beta_factor=.1,deletion_right_derivative=new_right_derivative,
        deleted_theta=float(deleted[0]),restored_theta=float(restored[0]),
        note="The parameter change was a pre-execution analytic rejection, not result-based tuning"))
    # Explicit failure contract at zero iteration budget: no false success.
    n=3; edges=np.array([[0,1],[0,2],[1,2]],dtype=np.int64); structure=EdgeObservationStructure(n,edges)
    stats=NodeStatistics.from_observations(structure,rng.normal(size=(10,n)),rng.normal(size=(10,n)),.8/n,.03)
    anchor=np.full(3,.3); zero=np.zeros(3)
    limited=_node_cd(stats.C,stats.h,stats.Gdiag,structure.edges,stats.eta**2/stats.sigma**2,
                     anchor,zero,zero,RHO,anchor,0,RELATIVE_KKT_TOLERANCE)
    assert limited[7] > limited[8] and limited[4] == 0
    checks.append(dict(case="zero_budget_nonconvergence_is_visible",passed=True))
    after={name:sha(root/name) for name in inherited}
    assert before == after
    result=dict(status="passed",scope="deterministic_algebra_only_no_development_no_confirmation_no_timing_campaign",
                diagnostic_rng_seed=9194831,compilation=compile_receipt,checks=checks,
                inherited_sources_sha256=after,inherited_sources_unchanged=True,
                source_sha256=sha(Path(__file__)))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=self_check()
        Path(__file__).with_name("CHECK_NODE_UPDATES.json").write_text(
            json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    else:
        parser.print_help()

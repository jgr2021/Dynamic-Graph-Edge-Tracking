"""Synthetic pressure and inventory benchmarks for the frozen finite-grid estimator."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

EDGES = np.array([(0, 1), (0, 2), (1, 3), (2, 3)])
SRC, DST = EDGES.T
LINK = np.arange(len(EDGES))
B = np.eye(4)[:, SRC] - np.eye(4)[:, DST]
STEPS = 48
BLOCKS = 32
SEEDS = range(112501, 112509)


def tank_block(rng, state, w):
    capacity = np.array([.8, 1., 1.2, 1.4])  # g/kPa
    dt, g_ref, sigma = .25, .4, .06  # s, g/(s kPa), kPa/sqrt(s)
    design, response = [], []
    for _ in range(STEPS):
        inflow = rng.normal(0., .55, 4)  # observed, signed external g/s
        old = state.copy()
        for _ in range(4):
            flow = g_ref * w * (B.T @ state)
            state += dt/4 * (inflow-B@flow)/capacity + sigma*np.sqrt(dt/4)*rng.normal(size=4)
        design.append((dt*g_ref/capacity[:, None])*B*(B.T@old))
        response.append(old+dt*inflow/capacity-state)
    return np.vstack(design), np.concatenate(response), sigma*np.sqrt(dt)


def shipment_block(rng, state, w, transit):
    design, response = [], []
    previous_order = np.zeros(4)
    for k in range(STEPS):
        order = rng.uniform(.1, 1.5, 4)  # commanded kg per route
        external = rng.normal(0., .3, 4)  # measured external net kg
        shipped = w * order
        before = state.copy()
        delta = external + rng.normal(0., .08, 4)
        np.add.at(delta, SRC, -shipped)
        np.add.at(delta, DST, transit)
        state += delta
        assert min(state)>0
        if k:  # arrivals at the first transition may use the previous block's weight
            H = np.zeros((4, 4))
            H[SRC, LINK] = -order
            H[DST, LINK] = previous_order
            design.append(H)
            response.append(state-before-external)
        transit[:] = shipped
        previous_order = order
    return np.vstack(design), np.concatenate(response), .08


def evaluate(r, model, task, seed):
    rng = np.random.default_rng(seed)
    initial_weights = rng.uniform(.35, .95, 4)
    changed = int(rng.integers(4))
    state = (200.+rng.normal(size=4)) if task=='tank' else np.full(4,5000.)
    transit = np.zeros(4)
    actor = r.OldActor(EDGES,model,paired=False)
    rows=[]
    for block in range(BLOCKS):
        w = initial_weights.copy()
        w[changed] = (.8,.3,0.,.8)[block//8]
        if task=='tank':
            H,y,sigma=tank_block(rng,state,w)
        else:
            H,y,sigma=shipment_block(rng,state,w,transit)
        assert np.linalg.matrix_rank(H) == len(w)
        posterior=actor.update(H.T@H/sigma**2,H.T@y/sigma**2)
        fit=lsq_linear(H,y,bounds=(0.,1.))
        assert fit.success
        rows.append(dict(task=task,seed=seed,block=block,changed=changed,truth=w.tolist(),
                         posterior=posterior['support_action'].tolist(),ls=fit.x.tolist(),
                         converged=bool(posterior['converged'])))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('code_root',type=Path,help='Directory containing theta_u_independent_20261027')
    args=parser.parse_args()
    sys.path.insert(0,str((args.code_root/'theta_u_independent_20261027').resolve()))
    import run as original
    r=original.r
    r.load_numeric()
    config=json.loads((original.V60/'PROTOCOL_V60.json').read_text(encoding='utf-8-sig'))
    model=r.model_load(config)
    rows=[]
    with threadpool_limits(limits=1):
        for task in ('tank','shipment'):
            for seed in SEEDS:
                rows.extend(evaluate(r,model,task,seed))
    summary={}
    for task in ('tank','shipment'):
        group=[x for x in rows if x['task']==task]
        summary[task]={'capped':sum(not x['converged'] for x in group)}
        for method in ('posterior','ls'):
            delta=np.array([np.subtract(x[method],x['truth']) for x in group])
            phases=[np.mean([x[method][x['changed']] for x in group if x['block']//8==phase]) for phase in range(4)]
            summary[task][method]={'rmse':np.sqrt(np.mean(delta**2)),'phase_means':phases}
    path=Path(__file__).resolve().parent
    (path/'application_results.json').write_text(json.dumps(dict(summary=summary,blocks=rows),indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))

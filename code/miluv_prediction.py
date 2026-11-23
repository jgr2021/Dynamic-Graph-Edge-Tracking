"""One-step altitude-prediction check on measured three-robot MILUV trajectories."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(next((ROOT / 'theta_u_topology_20261023').glob('*_v60'))))
import run_v60 as r

def altitude(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        fields = rows.fieldnames
        a, b = ('timestamp_ns', 'pz') if 'timestamp_ns' in fields else ('timestamp', 'pose.position.z')
        pairs = [(int(float(row[a])) if '.' in row[a] else int(row[a]), float(row[b])) for row in rows if row[b] and row[a]]
    pairs.sort()
    return np.array([x for x,z in pairs], dtype=np.int64), np.array([z for x,z in pairs])

def runs(folder):
    obs = [altitude(folder / robot / 'mocap.csv') for robot in ('ifo001','ifo002','ifo003')]
    dt = 100_000_000
    grid = np.arange(max(t[0] for t,z in obs), min(t[-1] for t,z in obs)+1, dt)
    nodes, valid = [], np.ones(len(grid),dtype=bool)
    for t,z in obs:
        idx = np.searchsorted(t,grid,side='right')-1
        nodes.append(z[idx])
        valid &= np.isfinite(z[idx]) & (grid-t[idx] < dt)
    height = np.stack(nodes,axis=1)
    for segment in np.split(np.flatnonzero(valid),np.flatnonzero(np.diff(np.flatnonzero(valid))>1)+1):
        if len(segment)>=97:
            yield height[segment]

def evaluate(folder, model):
    obj,B = r.structure(3)
    eta,sigma,K=.8/3,.03,48
    ss=np.zeros(3)
    count=0
    for z in runs(folder):
        actors=[r.OldActor(obj.edges,copy.deepcopy(model),paired=False),r.Author(obj.m,kind='hard',tau=.1,box=True)]
        plan=r.fast.IncidenceStatisticsPlan(obj)
        weights=None
        for start in range(0,len(z)-K,K):
            data=z[start:start+K+1]
            x=data[:-1]
            if weights is not None:
                for j,w in enumerate(weights):
                    L=(B*w)@B.T
                    diff=x-eta*x@L-data[1:]
                    ss[j]+=np.square(diff).sum()
                ss[2]+=np.square(x-data[1:]).sum()
                count+=K*3
            obs=dict(states=data[None,:,:],inputs=np.zeros((1,K,3)),a=1.,eta=eta,sigma=sigma)
            stat,X,y=r.statistics_for(obj,plan,obs,0)
            w=actors[0].update(stat.G,stat.h)['support_action'].copy()
            v,meta=actors[1].update(stat,dense_loader=r.dense_loader(obj,B,X,y,eta,sigma),compute_score=False)
            assert meta['success']
            weights=(w,v.copy())
    return dict(recording=folder.name, node_predictions=count, squared_error=ss.tolist())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('data',type=Path,help='Parent of recording/ifo00{1,2,3}/mocap.csv in the listed MILUV recordings')
    ap.add_argument('out',type=Path)
    args=ap.parse_args()
    names=json.loads((ROOT/'miluv_recordings.json').read_text())
    r.load_numeric()
    model=r.model_load(r.c.read(r.ROOT/'PROTOCOL_V60.json'))
    with threadpool_limits(limits=1):
        rows=[evaluate(args.data/n,model) for n in names]
    count=sum(row['node_predictions'] for row in rows)
    errors=np.sum([row['squared_error'] for row in rows],axis=0)
    result=dict(methods=['proposed','sparse_ekf','persistence'], rows=rows, node_predictions=count,rmse_metres=np.sqrt(errors/count).tolist())
    args.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(count,result['rmse_metres'])
if __name__=='__main__':
    main()


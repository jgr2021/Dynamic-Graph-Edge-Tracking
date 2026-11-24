"""Matched extensions to the published synthetic benchmark. Python 3.9 + original study code."""
import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

def independent_flips(belief, probability):
    """Apply the binary symmetric transition along every incident-edge axis."""
    answer = belief.copy()
    width = 1
    while width < answer.shape[-1]:
        pairs = answer.reshape(answer.shape[0], -1, 2, width)
        old = pairs.copy()
        pairs[:] = (1-probability)*old + probability*old[:, :, ::-1, :]
        width *= 2
    return answer

class BinaryTracker:
    """Local binary graph beliefs (Tenorio et al.), with block diffusion observations."""
    def __init__(self, B):
        n,m = B.shape
        self.indices = [np.flatnonzero(row) for row in B]
        self.states = np.array(list(itertools.product([0.,1.], repeat=n-1)))
        self.belief = np.full((n,len(self.states)),1/len(self.states))
        self.B = B

    def update(self,X,y,eta,sigma):
        prior = independent_flips(self.belief, .01)
        contrasts = eta*(X@self.B)
        marginal = np.zeros(self.B.shape[1])
        for node, idx in enumerate(self.indices):
            design = contrasts[:,idx]*self.B[node,idx]
            gram = design.T@design/(sigma*sigma)
            natural = design.T@y[:,node]/(sigma*sigma)
            score = self.states@natural - .5*np.sum((self.states@gram)*self.states,axis=1)
            logp = np.log(prior[node]) + score
            weights = np.exp(logp-logp.max())
            self.belief[node] = weights/weights.sum()
            marginal[idx] += .5*(self.belief[node]@self.states)
        return (marginal>.5).astype(float)

def select_shrinkage(original,r,out):
    # Separate graph and observation seeds; equal trajectory length and balanced conditions.
    grid=[0.,.001,.003,.01,.03,.1,.25]
    losses={tau:[] for tau in grid}
    for n,kind,sd in itertools.product((6,12),('bridge_failure','mobility_radius'),(0.,.3)):
        graph=original.gen.make_graph(n,kind,31050211,T=72)
        obs=original.gen.generate_observations(graph,41050211+10*n+int(sd*10),M=48,excitation=sd)
        obj,B=r.structure(n)
        plan=r.fast.IncidenceStatisticsPlan(obj)
        actors={tau:r.Author(obj.m,kind='soft',tau=tau,box=True) for tau in grid}
        for t,w in enumerate(graph['theta']):
            stat,X,y=r.statistics_for(obj,plan,obs,t)
            for tau,act in actors.items():
                p,meta=act.update(stat,dense_loader=r.dense_loader(obj,B,X,y,float(obs['eta']),float(obs['sigma'])))
                assert meta['success']
                losses[tau].append(float(np.linalg.norm(p-w)/max(1e-15,np.linalg.norm(w))))
    averages={tau:float(np.mean(v)) for tau,v in losses.items()}
    chosen=min(averages,key=averages.get)
    out.mkdir(exist_ok=True,parents=True)
    (out/'tuning.json').write_text(json.dumps(dict(mean_weight_error=averages,chosen=chosen),indent=2))
    return chosen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('outputs',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    archive=args.outputs/'theta_u_independent_20261027'
    sys.path.insert(0,str(archive))
    import run as original
    r=original.r
    r.load_numeric()
    config=json.loads((original.V60/'PROTOCOL_V60.json').read_text(encoding='utf-8-sig'))
    r.setup(config)
    chosen=select_shrinkage(original,r,args.destination)
    allrows=[]
    def evaluate(case,task,method,B,edges,w,points):
        prev=None
        for t,(truth,estimate) in enumerate(zip(w,points)):
            v=np.linalg.eigvalsh((B*truth)@B.T)
            ev=np.linalg.eigvalsh((B*estimate)@B.T)
            tc=original.components(len(B),edges,truth)
            ec=original.components(len(B),edges,estimate)
            event=t>0 and np.any((truth>0)!=(w[t-1]>0))
            err=np.mean(((estimate-prev)-(truth-w[t-1]))**2) if event else None
            allrows.append(dict(case=case,task=task,method=method,t=t,
                edge=float(np.linalg.norm(estimate-truth)/max(np.linalg.norm(truth),1e-15)),
                spectrum=float(np.linalg.norm(ev-v)/max(np.linalg.norm(v),1e-15)),
                disconnected=len(tc)>1,exact=ec==tc,split=len(ec)>1,event_mse=err))
            prev=estimate
    for path in sorted((archive/'runs').iterdir()):
        if not path.is_dir(): continue
        with np.load(path/'observations.npz') as f: obs=dict(f)
        n=obs['states'].shape[-1]
        obj,B=r.structure(n)
        with np.load(path/'truth.npz') as f: original_weights=f['theta'];edges=f['edges']
        assert np.array_equal(edges,obj.edges)
        with np.load(path/'inference.npz') as f:
            proposed=f['old_support__theta'];mean=f['old_mean__theta'];hard=f['hard_box__theta']
        for task in (['weighted','binary'] if n==6 else ['weighted']):
            data=obs.copy()
            truth=original_weights
            if task=='binary':
                # Paired simulation: retain original inputs, supports, initial state and realized innovations.
                truth=(truth>0).astype(float)
                a,eta=data['a'],data['eta']
                new=obs['states'].copy()
                for t,w in enumerate(truth):
                    old=obs['states'][t]
                    drift=a*np.eye(n)-eta*((B*original_weights[t])@B.T)
                    noise=old[1:]-old[:-1]@drift.T-obs['inputs'][t]
                    drift=a*np.eye(n)-eta*((B*w)@B.T)
                    if t and np.allclose(old[0],obs['states'][t-1,-1]):new[t,0]=new[t-1,-1]
                    for k,u in enumerate(data['inputs'][t]):
                        new[t,k+1]=drift@new[t,k]+u+noise[k]
                data['states']=new
            plan=r.fast.IncidenceStatisticsPlan(obj)
            actors={name:r.Author(obj.m,kind=kind,tau=tau,box=True) for name,kind,tau in
                    [('soft_dev','soft',chosen),('soft025','soft',.25),('soft010','soft',.1),('hard','hard',.1)]}
            tracker=BinaryTracker(B)
            actor=r.OldActor(obj.edges,copy.deepcopy(r.model_load(config)),paired=False) if task=='binary' else None
            points={key:[] for key in [*actors,'binary_marginal','proposed','mean_cut010']}
            for t in range(len(truth)):
                stat,X,y=r.statistics_for(obj,plan,data,t)
                sigma=float(data['sigma']);eta=float(data['eta'])
                for name,baseline in actors.items():
                    point,meta=baseline.update(stat,dense_loader=r.dense_loader(obj,B,X,y,eta,sigma))
                    assert meta['success']
                    points[name].append(point.copy())
                points['binary_marginal'].append(tracker.update(X,y,eta,sigma))
                if actor:
                    result=actor.update(stat.G,stat.h)
                    p,mu=result['support_action'],result['mean_action']
                else:
                    p,mu=proposed[t],mean[t]
                    np.testing.assert_allclose(points['hard'][-1],hard[t],atol=1e-9)
                points['proposed'].append(p.copy())
                points['mean_cut010'].append(np.where(mu>.1,mu,0.))
            for name,p in points.items():
                evaluate(path.name,task,name,B,edges,truth,p)
        print(path.name,flush=True)
    summary={}
    for task in ['weighted','binary']:
        summary[task]={}
        for method in sorted(set(row['method'] for row in allrows)):
            rows=[row for row in allrows if row['task']==task and row['method']==method]
            assert len(rows)==(1728 if task=='weighted' else 864)
            off=[row for row in rows if row['disconnected']]
            on=[row for row in rows if not row['disconnected']]
            summary[task][method]=dict(edge=np.mean([row['edge'] for row in rows]),
                spectrum=np.mean([row['spectrum'] for row in rows]),
                correct=sum(row['exact'] for row in off),off=len(off),
                splits=sum(row['split'] for row in on),on=len(on),
                event_rmse=np.sqrt(np.mean([row['event_mse'] for row in rows if row['event_mse'] is not None])))
    args.destination.mkdir(exist_ok=True,parents=True)
    (args.destination/'scores.json').write_text(json.dumps(summary,indent=2))
    with (args.destination/'blocks.jsonl').open('w') as f:
        for row in allrows:f.write(json.dumps(row)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()


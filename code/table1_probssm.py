"""Log-domain reproduction of the two ProbSSM readouts on Table I observations."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'theta_u_independent_20261027'))
from run import components
class StableBlock:
 def __init__(self,B,sigma):
  self.B=B;self.n=B.shape[0];self.edges=[tuple(np.flatnonzero(B[:,e])) for e in range(B.shape[1])];self.sigma=sigma
  self.inc=[[e for e,p in enumerate(self.edges) if node in p] for node in range(self.n)]
  self.states=[np.asarray(list(itertools.product((0.,1.),repeat=len(inc))),dtype=float) for inc in self.inc]
  self.logb=[np.full(len(st),-np.log(len(st))) for st in self.states]
 def update(self,Q,Y):
  # Q shape (M,n), Y shape (M,n). Mix once per block, multiply independent likelihoods.
  D=Q@self.B;H=D[:,:,None]*self.B.T[None,:,:]; ps=[];maps=[]
  for n in range(self.n):
   st=self.states[n];prior=np.logaddexp(np.log(.99)+self.logb[n],np.log(.01/len(st)))
   hn=H[:,self.inc[n],n];yhat=st@hn.T;z=(Y[:,n][None,:]-yhat)/self.sigma
   logw=prior-.5*np.sum(z*z,axis=1);self.logb[n]=logw-logsumexp(logw)
   b=np.exp(self.logb[n]);ps.append(b@st);maps.append(st[np.argmax(self.logb[n])])
  avg=np.empty(len(self.edges));mode=np.empty_like(avg)
  for e,(i,j) in enumerate(self.edges):
   a=self.inc[i].index(e);b=self.inc[j].index(e);avg[e]=(ps[i][a]+ps[j][b])/2;mode[e]=(maps[i][a]+maps[j][b])/2
  return avg,mode


def evaluate(directory):
    with np.load(directory/'observations.npz') as raw:
        states,inputs,a,eta,sigma=(raw[k] for k in ('states','inputs','a','eta','sigma'))
    n=states.shape[-1]
    edges=np.array(list(itertools.combinations(range(n),2)))
    B=np.zeros((n,len(edges)))
    B[edges[:,0],np.arange(len(edges))]=1
    B[edges[:,1],np.arange(len(edges))]=-1
    actor=StableBlock(B,sigma)
    estimates=[]
    for s,u in zip(states,inputs):
        estimates.append(actor.update(eta*s[:-1],a*s[:-1]+u-s[1:]))
    rows=[]
    with np.load(directory/'truth.npz') as data:
        truth=data['theta']
    for (avg,mode),w in zip(estimates,truth):
        lamb=np.linalg.eigvalsh((B*w)@B.T)
        partition=components(n,edges,w)
        for method,point in [('average',avg),('local_map',mode)]:
            vals=np.linalg.eigvalsh((B*point)@B.T)
            est=components(n,edges,point)
            rows.append((method,np.linalg.norm(w-point)/max(1e-15,np.linalg.norm(w)),np.linalg.norm(lamb-vals)/max(1e-15,np.linalg.norm(lamb)),len(partition)>1,partition==est,len(partition)==1 and len(est)>1))
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,default=ROOT/'theta_u_independent_20261027/runs')
    args=parser.parse_args()
    rows=[]
    with threadpool_limits(limits=1):
        for d in sorted(args.runs.iterdir()):
            if d.is_dir():
                rows.extend(evaluate(d))
    if len(rows)!=2*24*72:
        raise ValueError('Expected 24 complete 72-block trajectories')
    for method in ('average','local_map'):
        rr=[r for r in rows if r[0]==method]
        off=[r for r in rr if r[3]]
        print(method,dict(edge=np.mean([r[1] for r in rr]),eigenvalue=np.mean([r[2] for r in rr]),partitions=sum(r[4] for r in off),false_splits=sum(r[5] for r in rr)))
if __name__=='__main__':
    main()

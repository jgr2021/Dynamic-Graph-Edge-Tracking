import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import pathlib,sys,time,json,itertools,hashlib
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT.parent; PRIOR=OUT/'theta_u_independent_20261027'; PROJECT=OUT/'theta_u_topology_20261023'
SOURCE=PROJECT/'baseline_polyekf_source_20261025'/'raw_snapshot'
sys.path.insert(0,str(SOURCE));from baseline_prob_ssm import ProbSSMBaseline

def write(path,obj):
 with (ROOT/path).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def incidence(n):
 edges=np.array(list(itertools.combinations(range(n),2)),dtype=int);B=np.zeros((n,len(edges)));B[edges[:,0],np.arange(len(edges))]=1.;B[edges[:,1],np.arange(len(edges))]=-1.;return B,edges

class AuditedAuthor(ProbSSMBaseline):
 def __init__(self,B,sigma):
  super().__init__(B,poly_c=np.array([0.,1.]),p_e=.01,sigma=sigma,w_ref=1.,state_est='avg');self.zero_resets=0;self._active_node=0
 def __call__(self,*args,**kwargs):self._active_node=0;return super().__call__(*args,**kwargs)
 def _gaussian_pdf(self,x,mu,sigma):
  likelihood=super()._gaussian_pdf(x,mu,sigma);i=self._active_node;self._active_node+=1;b=self._beliefs[i];prior=(1.-self.p_e)*b+self.p_e/len(b)
  self.zero_resets+=int(float(np.sum(prior*likelihood))==0.)
  return likelihood
 def outputs(self):
  pairs=self.edge_pairs;out=[]
  for mode in ('avg','map'):
   self.state_est=mode;out.append(np.array([(self._marginal(i,e)+self._marginal(j,e))/2 for e,(i,j) in enumerate(pairs)]))
  self.state_est='avg';return out

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

def precheck():
 B,_=incidence(4);raw=AuditedAuthor(B,.7);stable=StableBlock(B,.7);err=0.;rng=np.random.default_rng(927471)
 for block in range(4):
  Q=rng.normal(0,.08,(5,4));Y=rng.normal(0,.03,(5,4))
  for k in range(len(Q)):raw.p_e=.01 if k==0 else 0.;raw(Q[k],Y[k])
  a,b=raw.outputs();c,d=stable.update(Q,Y)
  err=max(err,float(np.max(abs(a-c))),float(np.max(abs(b-d))),max(float(np.max(abs(p-np.exp(q)))) for p,q in zip(raw._beliefs,stable.logb)))
 if err>1e-10 or raw.zero_resets:raise AssertionError('author/stable artificial equivalence failed')
 return {'passed':True,'blocks':4,'max_abs_error':err,'author_underflow_resets':raw.zero_resets,'interpretation':'equivalence only for these artificial inputs, not real data'}

def main():
 with threadpool_limits(limits=1):
  pre=precheck();write('PRECHECK.json',pre);print('PRECHECK',pre,flush=True)
  original=json.loads((PRIOR/'SUMMARY.json').read_text(encoding='utf8'));cases=original['cases']; all_cost={};underflow={};tstart=time.monotonic()
  for idx,cid in enumerate(cases):
   if time.monotonic()-tstart>7100:raise TimeoutError('batch time limit')
   with np.load(PRIOR/'runs'/cid/'observations.npz',allow_pickle=False) as f:obs={k:np.array(f[k]) for k in ('states','inputs','a','eta','sigma')}
   states=obs['states'];inputs=obs['inputs'];n=states.shape[-1]; B,_=incidence(n);a=float(obs['a']);eta=float(obs['eta']);sigma=float(obs['sigma'])
   clocks={};start=time.perf_counter();raw=AuditedAuthor(B,sigma);clocks['raw_init']=time.perf_counter()-start
   start=time.perf_counter();stable=StableBlock(B,sigma);clocks['stable_init']=time.perf_counter()-start
   vals={k:[] for k in ('raw_avg','raw_map','stable_avg','stable_map')};U={k:[] for k in vals};ev={k:[] for k in vals};cost={k:0. for k in vals};diff=[];reset_perblock=[]
   for t in range(72):
    start=time.perf_counter();Q=eta*states[t,:-1];Y=a*states[t,:-1]+inputs[t]-states[t,1:];data_seconds=time.perf_counter()-start
    before=raw.zero_resets;start=time.perf_counter()
    for k in range(48):raw.p_e=.01 if k==0 else 0.;raw(Q[k],Y[k])
    avg,mode=raw.outputs();base=time.perf_counter()-start+data_seconds;reset_perblock.append(raw.zero_resets-before)
    start=time.perf_counter();savg,smode=stable.update(Q,Y);stablecost=time.perf_counter()-start+data_seconds
    diff.append([float(np.max(np.abs(avg-savg))),float(np.max(np.abs(mode-smode)))])
    for name,point,core in [('raw_avg',avg,base),('raw_map',mode,base),('stable_avg',savg,stablecost),('stable_map',smode,stablecost)]:
     start=time.perf_counter();L=(B*point)@B.T;v,vec=np.linalg.eigh(L);cost[name]+=core+time.perf_counter()-start
     vals[name].append(point);ev[name].append(v);U[name].append(vec)
   path=ROOT/'runs'/cid;path.mkdir(parents=True,exist_ok=False)
   np.savez_compressed(path/'INFERENCE.npz',**{k+'__theta':np.array(vals[k]) for k in vals},**{k+'__lambda':np.array(ev[k]) for k in vals},**{k+'__U':np.array(U[k]) for k in vals},raw_reset_counts=np.array(reset_perblock),raw_stable_maxabs=np.array(diff))
   all_cost[cid]=dict(online=cost,initialization=clocks);underflow[cid]=int(raw.zero_resets)
   print(json.dumps({'done':idx+1,'case':cid,'resets':underflow[cid],'seconds':round(time.monotonic()-tstart,2)}),flush=True)
  write('INFERENCE_COMPLETE.json',{'cases':cases,'case_count':len(cases),'costs':all_cost,'underflow':underflow,'true_fields_and_existing_results_read':False,'wall_seconds':time.monotonic()-tstart})
if __name__=='__main__':main()

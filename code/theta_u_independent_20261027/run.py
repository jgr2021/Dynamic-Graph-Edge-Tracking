import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import sys,time,json,copy,hashlib,importlib.util,itertools,pathlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=ROOT.parent/'theta_u_topology_20261023'; V60=PROJECT/'continuation_20261027_persistent_jump_v60'
sys.path.insert(0,str(V60)); import run_v60 as r
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fresh(name,obj):
 with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
spec=importlib.util.spec_from_file_location('heldout_generator',PROJECT/'model.py'); gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
def components(n,edges,w):
 parent=list(range(n))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for (i,j),v in zip(edges,w):
  if v>0:parent[find(int(i))]=find(int(j))
 return tuple(sorted(tuple(i for i in range(n) if find(i)==z) for z in set(find(i) for i in range(n))))
def metrics(B,edges,point,truth,U,val,Ltrue,Utrue,vtrue):
 n=B.shape[0]; L=(B*point)@B.T
 den=lambda a,b:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-15))
 assert np.linalg.norm(L@U-U*val)<1e-7 and np.linalg.norm(U.T@U-np.eye(n))<1e-7
 pg=[]
 for k in range(1,n):
  gap=min(vtrue[k]-vtrue[k-1],vtrue[k+1]-vtrue[k] if k+1<n else float('inf'))
  egap=min(val[k]-val[k-1],val[k+1]-val[k] if k+1<n else float('inf'))
  if gap>1e-6 and egap>1e-6:pg.append(float(np.linalg.norm(np.outer(U[:,k],U[:,k])-np.outer(Utrue[:,k],Utrue[:,k]))))
 tc=components(n,edges,truth);pc=components(n,edges,point)
 return dict(theta=den(point,truth),L=den(L,Ltrue),lam=den(val,vtrue),proj_sum=sum(pg),proj_count=len(pg),true_components=len(tc),partition_correct=pc==tc,false_split=(len(tc)==1 and len(pc)>1))

def main():
 with threadpool_limits(limits=1):
  t0=time.monotonic();pro=json.loads((V60/'PROTOCOL_V60.json').read_text(encoding='utf-8-sig'))
  r.load_numeric();r.setup(pro)
  methods=['old_support','old_mean','fast_support','fast_mean','reset_support','gaussian','hard_box']
  rows=[];cases=[]; diagnostics=[]; elapsed={x:0. for x in methods}; init_cost={x:0. for x in methods};caps={'old':0,'fast':0,'reset':0};
  for n,ki,ei,si in itertools.product((6,12),range(2),range(2),range(3)):
   if time.monotonic()-t0>7100:raise TimeoutError('Budget')
   kind=('bridge_failure','mobility_radius')[ki];exc=(0.,.3)[ei];gs=19876100+si;obsseed=19886100+100*n+20*ki+5*ei+si; cid=f'n{n}_{kind}_e{ei}_g{gs}_o{obsseed}'
   graph=gen.make_graph(n,kind,gs,T=72);ob=gen.generate_observations(graph,obsseed,M=48,excitation=exc)
   observed={k:ob[k] for k in ('states','inputs','a','eta','sigma')}; nedge=len(graph['edges'])
   outdir=ROOT/'runs'/cid;outdir.mkdir(parents=True,exist_ok=False)
   np.savez_compressed(outdir/'observations.npz',**observed);np.savez_compressed(outdir/'truth.npz',theta=graph['theta'],edges=graph['edges'])
   # Point estimates are computed without reading truth or graph fields.
   obj,B=r.structure(n);plans={x:r.fast.IncidenceStatisticsPlan(obj) for x in ('old','fast','reset','gaussian','hard')}
   model=r.model_load(pro);ctors={'old':lambda:r.OldActor(obj.edges,copy.deepcopy(model),paired=False),'fast':lambda:r.kernel_actor.Actor(obj.m,r.kernel_actor.make_model(copy.deepcopy(model)),mode='fast'),'gaussian':lambda:r.Gaussian(obj.m),'hard':lambda:r.Author(obj.m,kind='hard',tau=.1,box=True)}
   actors={};
   for name in ('old','fast','gaussian','hard'):
    clock=time.perf_counter();actors[name]=ctors[name]();dt=time.perf_counter()-clock
    names={'old':['old_support','old_mean'],'fast':['fast_support','fast_mean'],'gaussian':['gaussian'],'hard':['hard_box']}[name]
    for nm in names:init_cost[nm]+=dt
   estimates={x:[] for x in methods}; evals={x:[] for x in methods}; evecs={x:[] for x in methods}
   for t in range(72):
    for name in ('old','fast','reset','gaussian','hard'):
     clock=time.perf_counter();stat,X,response=r.statistics_for(obj,plans[name],observed,t)
     if name=='reset':actor=ctors['old']()
     else:actor=actors[name]
     if name in ('old','fast','reset'):
      result=actor.update(stat.G,stat.h);caps[name]+=int(not result['converged']);pts={name+'_support':result['support_action']}
      if name in ('old','fast'):pts[name+'_mean']=result['mean_action']
     elif name=='gaussian':
      point,meta=actor.update(stat)
      if not meta['success']:raise RuntimeError('gaussian failure')
      pts={'gaussian':point}
     else:
      point,meta=actor.update(stat,dense_loader=r.dense_loader(obj,B,X,response,float(observed['eta']),float(observed['sigma'])),compute_score=False)
      if not meta['success']:raise RuntimeError('paper adaptation failure')
      pts={'hard_box':point}
     core=time.perf_counter()-clock
     for nm,point in pts.items():
      clock=time.perf_counter();L=(B*point)@B.T;val,U=np.linalg.eigh(L);cost=time.perf_counter()-clock
      elapsed[nm]+=core+cost;estimates[nm].append(np.array(point));evals[nm].append(val);evecs[nm].append(U)
   np.savez_compressed(outdir/'inference.npz',**{nm+'__theta':np.array(estimates[nm]) for nm in methods},**{nm+'__values':np.array(evals[nm]) for nm in methods},**{nm+'__U':np.array(evecs[nm]) for nm in methods})
   # Evaluation only after every method for this complete trajectory has finished.
   with np.load(outdir/'truth.npz') as z:truth=z['theta'];edges=z['edges']
   for t in range(72):
    Lt=(B*truth[t])@B.T;vtrue,Utrue=np.linalg.eigh(Lt)
    for nm in methods:
     point=estimates[nm][t];rec=metrics(B,edges,point,truth[t],evecs[nm][t],evals[nm][t],Lt,Utrue,vtrue)
     rec.update(case=cid,t=t,method=nm,n=n,kind=kind,excitation=exc,graph_seed=gs,obs_seed=obsseed)
     if t:
      rec['delta_sq']=float(np.mean(((point-estimates[nm][t-1])-(truth[t]-truth[t-1]))**2))
      rec['event']=bool(np.any((truth[t]>0)!=(truth[t-1]>0)))
     else:rec['delta_sq']=None;rec['event']=False
     rows.append(rec)
   cases.append(cid);print(json.dumps({'completed':cid,'count':len(cases),'elapsed':round(time.monotonic()-t0,2)}),flush=True)
  fresh('ROWS.json',rows);summary={}
  for nm in methods:
   rr=[x for x in rows if x['method']==nm]; off=[x for x in rr if x['true_components']>1];on=[x for x in rr if x['true_components']==1]; events=[x['delta_sq'] for x in rr if x['event']]; percase=[]
   for cid in cases:
    subset=[x for x in rr if x['case']==cid];percase.append(dict(case=cid,theta=float(np.mean([x['theta'] for x in subset])),L=float(np.mean([x['L'] for x in subset])),lam=float(np.mean([x['lam'] for x in subset]))))
   summary[nm]=dict(theta=float(np.mean([x['theta'] for x in rr])),L=float(np.mean([x['L'] for x in rr])),lam=float(np.mean([x['lam'] for x in rr])),projector_mean=sum(x['proj_sum'] for x in rr)/max(1,sum(x['proj_count'] for x in rr)),projector_count=sum(x['proj_count'] for x in rr),off_correct=sum(x['partition_correct'] for x in off),off_total=len(off),false_split=sum(x['false_split'] for x in on),on_total=len(on),event_delta_rmse=float(np.sqrt(np.mean(events))) if events else None,seconds=elapsed[nm],init_seconds=init_cost[nm],per_case=percase)
  fresh('SUMMARY.json',dict(cases=cases,summary=summary,caps=caps,seconds=time.monotonic()-t0,limitations=['new draws from existing graph families','full synchronous observations','paper hard-box adaptation','common truth-matched single eigenvalue eligibility differs by method','timing not isolated machine','seed graph reuse across condition panels','no separate U recursion']))
  print('COMPLETED',flush=True)
if __name__=='__main__':main()

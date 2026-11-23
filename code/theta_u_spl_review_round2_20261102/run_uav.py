import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import sys,json,time,copy,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
V60=BASE/'theta_u_topology_20261023/continuation_20261027_persistent_jump_v60'
sys.path.insert(0,str(V60))
import run_v60 as r

def dump(p,o):
 with p.open('x',encoding='utf-8') as f: json.dump(o,f,indent=2,allow_nan=False)

def main():
 protocol=json.loads((ROOT/'UAV_PROTOCOL.json').read_text())
 out=ROOT/'uav_run';out.mkdir(exist_ok=False)
 start=time.monotonic();n=6;T=36;K=48;a=.95;eta=.8/n;sigma=.03
 with threadpool_limits(limits=1):
  r.load_numeric()
  pro=json.loads((V60/'PROTOCOL_V60.json').read_text(encoding='utf-8-sig'));r.setup(pro)
  obj,B=r.structure(n);edges=np.asarray(obj.edges);model=r.model_load(pro)
  # Verify the forward/regression identity independently of any scored trajectory.
  test=np.arange(n)*.1;w=np.linspace(0,1,obj.m);L=(B*w)@B.T
  H=eta*B*(B.T@test);assert np.max(abs(H@w-eta*L@test))<1e-12
  rows=[];caps=0;case_summary=[]
  for seed in protocol['seeds']:
   for excitation in protocol['excitation']:
    if time.monotonic()-start>1700: raise TimeoutError('finite batch budget')
    cid=f'g{seed}_e{excitation}';folder=out/cid;folder.mkdir()
    gr=np.random.default_rng(seed);rg=np.random.default_rng(seed+100000+int(excitation*1000))
    center=gr.uniform(-22,22,(n,2));phase=gr.uniform(0,2*np.pi,(n,2));rate=gr.uniform(.04,.10,(n,2))
    positions=np.array([center+8*np.sin(phase+rate*t) for t in range(T)])
    distances=np.linalg.norm(positions[:,edges[:,0]]-positions[:,edges[:,1]],axis=2)
    weights=np.where(distances<35,np.exp(-(distances/25)**2),0.)
    states=[];inputs=[];truthL=[];s=rg.normal(0,.5,n)
    for t in range(T):
     Lt=(B*weights[t])@B.T;truthL.append(Lt);F=a*np.eye(n)-eta*Lt
     assert np.max(abs(np.linalg.eigvalsh(F)))<=.95000001
     block=[s.copy()];u=rg.normal(0,excitation,(K,n))
     for k in range(K):
      s=F@s+u[k]+rg.normal(0,sigma,n);block.append(s.copy())
     states.append(block);inputs.append(u)
    obs=dict(states=np.array(states),inputs=np.array(inputs),a=a,eta=eta,sigma=sigma)
    np.savez_compressed(folder/'observations.npz',**obs)
    np.savez_compressed(folder/'truth.npz',theta=weights,positions=positions,edges=edges)
    # Inference receives only node states and known inputs/parameters.
    actors={'proposed':r.OldActor(obj.edges,copy.deepcopy(model),paired=False),'sparse_ekf':r.Author(obj.m,kind='hard',tau=.1,box=True)}
    plans={m:r.fast.IncidenceStatisticsPlan(obj) for m in actors}
    saved={m:{k:[] for k in ['theta','values','U','seconds']} for m in actors}
    for t in range(T):
     for m in (list(actors) if t%2==0 else list(actors)[::-1]):
      clk=time.perf_counter();stat,X,y=r.statistics_for(obj,plans[m],obs,t)
      if m=='proposed':
       result=actors[m].update(stat.G,stat.h);point=result['support_action'];caps+=int(not result['converged'])
      else:
       point,meta=actors[m].update(stat,dense_loader=r.dense_loader(obj,B,X,y,eta,sigma),compute_score=False)
       assert meta['success']
      Lhat=(B*point)@B.T;v,U=np.linalg.eigh(Lhat);elapsed=time.perf_counter()-clk
      assert np.linalg.norm(Lhat@U-U*v)<1e-8 and np.linalg.norm(U.T@U-np.eye(n))<1e-8
      for k,val in [('theta',point),('values',v),('U',U),('seconds',elapsed)]:saved[m][k].append(np.array(val).copy())
    np.savez_compressed(folder/'inference.npz',**{m+'__'+k:np.array(v) for m,s in saved.items() for k,v in s.items()})
    # Truth-scoring starts only after both complete trajectory estimates are saved.
    rng_eval=np.random.default_rng(seed+200000+int(excitation*1000));cr=[]
    for t in range(T):
     Lt=truthL[t];vt=np.linalg.eigvalsh(Lt);F=a*np.eye(n)-eta*Lt;s=obs['states'][t,-1];wind=rng_eval.normal(0,sigma,n)
     allF={'no_graph':a*np.eye(n),'true_graph':F}
     for m in actors:
      point=saved[m]['theta'][t];v=saved[m]['values'][t];U=saved[m]['U'][t]
      allF[m]=(U*(a-eta*v))@U.T
      rec=dict(case=cid,seed=seed,excitation=excitation,t=t,method=m,edge=float(np.linalg.norm(point-weights[t])/max(np.linalg.norm(weights[t]),1e-15)),spectrum=float(np.linalg.norm(v-vt)/max(np.linalg.norm(vt),1e-15)),seconds=float(saved[m]['seconds'][t]))
      rows.append(rec);cr.append(rec)
     for m,Fhat in allF.items():
      control=np.clip(-Fhat@s/2.,-1.,1.)
      nxt=F@s+control+wind;cost=float(nxt@nxt+control@control)
      rows.append(dict(case=cid,seed=seed,excitation=excitation,t=t,method=m,decision_cost=cost))
    case_summary.append(dict(case=cid,seed=seed,excitation=excitation,**{m:{'edge':float(np.mean([v['edge'] for v in cr if v['method']==m])),'spectrum':float(np.mean([v['spectrum'] for v in cr if v['method']==m]))} for m in actors}))
    print(cid,'completed',round(time.monotonic()-start,2),flush=True)
  summary={}
  for excitation in protocol['excitation']:
   panel={}
   for m in ['proposed','sparse_ekf','no_graph','true_graph']:
    rr=[v for v in rows if v['method']==m and v['excitation']==excitation]
    d={}
    for key in ['edge','spectrum','seconds','decision_cost']:
     values=[v[key] for v in rr if key in v]
     if values:d[key]=float(np.mean(values))
    panel[m]=d
   summary[str(excitation)]=panel
  dump(out/'ROWS.json',rows);dump(out/'SUMMARY.json',dict(summary=summary,cases=case_summary,caps=caps,wall_seconds=time.monotonic()-start,trajectories=12,blocks=432,qualification='synthetic high-level altitude coordination; one-step matched-state decision replay, not closed-loop flight; full EVD'))
  sources={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(BASE/'theta_u_topology_20261023') in str(Path(m.__file__).resolve()) and Path(m.__file__).is_file()}
  dump(out/'SOURCE_HASHES.json',sources)

if __name__=='__main__':main()

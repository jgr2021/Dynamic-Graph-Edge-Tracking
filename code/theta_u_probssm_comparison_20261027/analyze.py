import pathlib,json,hashlib,itertools
import numpy as np
root=pathlib.Path(__file__).resolve().parent;prior=root.parent/'theta_u_independent_20261027'
def read(p):return json.loads(p.read_text(encoding='utf8'))
complete=read(root/'INFERENCE_COMPLETE.json');old=read(prior/'SUMMARY.json')
assert complete['true_fields_and_existing_results_read'] is False and complete['cases']==old['cases']
new=['raw_avg','raw_map','stable_avg','stable_map'];oldnames=['old_support','fast_support','gaussian','hard_box'];allnames=oldnames+new
def components(n,edges,w):
 parents=list(range(n))
 def find(x):
  while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
  return x
 for (i,j),v in zip(edges,w):
  if v>0:parents[find(int(i))]=find(int(j))
 return tuple(sorted(tuple(i for i in range(n) if find(i)==z) for z in set(find(i) for i in range(n))))
def gap(v,k):return min(v[k]-v[k-1],v[k+1]-v[k] if k+1<len(v) else float('inf'))
records=[];common={k:[] for k in allnames};res=orth=diffavg=diffmap=0.;reset=0;percase=[]
for cid in complete['cases']:
 with np.load(prior/'runs'/cid/'truth.npz',allow_pickle=False) as z: truth=z['theta'];edges=z['edges']
 with np.load(prior/'runs'/cid/'inference.npz',allow_pickle=False) as z:olddata={k:np.array(z[k]) for k in z.files}
 with np.load(root/'runs'/cid/'INFERENCE.npz',allow_pickle=False) as z:newdata={k:np.array(z[k]) for k in z.files}
 n=int(np.max(edges))+1;B=np.zeros((n,len(edges)));B[edges[:,0],np.arange(len(edges))]=1;B[edges[:,1],np.arange(len(edges))]=-1
 reset+=int(np.sum(newdata['raw_reset_counts']));diffavg=max(diffavg,float(np.max(newdata['raw_stable_maxabs'][:,0])));diffmap=max(diffmap,float(np.max(newdata['raw_stable_maxabs'][:,1])))
 for t in range(72):
  Ltrue=(B*truth[t])@B.T;vtrue,Utrue=np.linalg.eigh(Ltrue);tc=components(n,edges,truth[t]); vals={};evec={}
  for name in allnames:
   d=olddata if name in oldnames else newdata;point=d[name+'__theta'][t];v=d[name+('__values' if name in oldnames else '__lambda')][t];U=d[name+'__U'][t]
   L=(B*point)@B.T;res=max(res,float(np.linalg.norm(L@U-U*v)));orth=max(orth,float(np.linalg.norm(U.T@U-np.eye(n))));vals[name]=v;evec[name]=U
   den=lambda x,y:float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-15))
   pc=components(n,edges,point);event=bool(t and np.any((truth[t]>0)!=(truth[t-1]>0)))
   delta=float(np.mean(((point-d[name+'__theta'][t-1])-(truth[t]-truth[t-1]))**2)) if t else None
   records.append(dict(case=cid,t=t,n=n,method=name,theta=den(point,truth[t]),L=den(L,Ltrue),lam=den(v,vtrue),off=len(tc)>1,correct=pc==tc,false_split=len(tc)==1 and len(pc)>1,event=event,delta_sq=delta))
  for k in range(1,n):
   if gap(vtrue,k)<=1e-6 or any(gap(vals[name],k)<=1e-6 for name in allnames):continue
   for name in allnames:
    u=evec[name][:,k];u0=Utrue[:,k];common[name].append(float(np.linalg.norm(np.outer(u,u)-np.outer(u0,u0))))
summary={}
for name in allnames:
 rr=[r for r in records if r['method']==name];off=[r for r in rr if r['off']];on=[r for r in rr if not r['off']];events=[r['delta_sq'] for r in rr if r['event']]
 summary[name]=dict(theta=float(np.mean([r['theta'] for r in rr])),L=float(np.mean([r['L'] for r in rr])),lam=float(np.mean([r['lam'] for r in rr])),off_correct=sum(r['correct'] for r in off),off_total=len(off),false_split=sum(r['false_split'] for r in on),on_total=len(on),event_delta_rmse=float(np.sqrt(np.mean(events))),common_projector=float(np.mean(common[name])) if common[name] else None,common_count=len(common[name]),by_n={str(n):float(np.mean([r['theta'] for r in rr if r['n']==n])) for n in (6,12)},percase={cid:float(np.mean([r['theta'] for r in rr if r['case']==cid])) for cid in complete['cases']})
newcost={k:sum(v['online'][k] for v in complete['costs'].values()) for k in new};init={k:sum(v['initialization'][k] for v in complete['costs'].values()) for k in ('raw_init','stable_init')}
assert res<1e-7 and orth<1e-7 and len(records)==24*72*len(allnames)
checks=dict(spectral_checks=len(records),max_spectral_residual=res,max_orthogonality=orth,common_clusters=len(common[allnames[0]]),raw_stable_avg_maxabs=diffavg,raw_stable_map_maxabs=diffmap,author_zero_resets=reset,precheck=read(root/'PRECHECK.json'))
def write(name,obj):
 with (root/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
write('ROWS.json',records);write('AUDIT_SUMMARY.json',dict(summary=summary,new_online_seconds=newcost,new_initialization_seconds=init,checks=checks,limitations=['Previously exposed data','ProbSSM fixed positive amplitude 1','Original source and stable adaptation differ in floating point operation order','Prior method timing not same batch','No complete posterior or old/current covariance','Common projector posthoc common eligibility']))
print(json.dumps({'summary':{k:{x:v for x,v in q.items() if x!='percase'} for k,q in summary.items()},'new_online_seconds':newcost,'new_initialization_seconds':init,'checks':checks},ensure_ascii=False,indent=2))

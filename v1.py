import numpy as np,pandas as pd,yfinance as yf,matplotlib.pyplot as plt;from numba import njit,prange;from sklearn.preprocessing import StandardScaler as SS;from sklearn.linear_model import LogisticRegression as LR,Ridge as RD;from sklearn.metrics import accuracy_score as AC,r2_score as RS;from catboost import CatBoostClassifier as CBC,CatBoostRegressor as CBR;from lightgbm import LGBMClassifier as LC,LGBMRegressor as LGR;from xgboost import XGBClassifier as XC,XGBRegressor as XR;import warnings;warnings.filterwarnings('ignore')
yf.enable_debug_mode()
@njit(fastmath=True)
def vg(s):
 N=len(s);ki,ko=np.zeros(N),np.zeros(N)
 for i in range(N-1):
  m=-np.inf
  for j in range(i+1,N):
   p=(s[j]-s[i])/(j-i)
   if p>m:ko[i]+=1;ki[j]+=1;m=p
 return ki,ko,ki+ko
@njit(fastmath=True)
def kld(ki,ko):
 N=len(ki);mx=int(max(np.max(ki),np.max(ko)))+1;ci,co=np.zeros(mx),np.zeros(mx)
 for i in range(N):ci[int(ki[i])]+=1;co[int(ko[i])]+=1
 pi,po=(ci/N)+(1/N),(co/N)+(1/N);pi/=np.sum(pi);po/=np.sum(po);return np.sum(pi*np.log(pi/po))
@njit(fastmath=True)
def hs(kt):
 c=np.zeros(int(np.max(kt))+1)
 for v in kt:c[int(v)]+=1
 k=np.nonzero(c)[0];p=c[k]/len(kt);m=(k>=2)&(p>0)
 if np.sum(m)<3:return 0.5
 x,y=np.log(k[m].astype(np.float64)),np.log(p[m]);n=len(x);sx=np.sum(x)
 return max(0.01,min(0.99,(3+(n*np.sum(x*y)-sx*np.sum(y))/(n*np.sum(x*x)-sx**2))/2))
@njit(fastmath=True)
def fvg(s):
 ki,ko,kt=vg(s);W=len(s);nb,nt=max(3,W//10),min(10,W//3);gr,ga=np.mean(kt[-nb:]),max(1.,np.mean(kt[:nb]));sl=0.
 if nt>=3:i=np.arange(nt,dtype=np.float64);sl=(nt*np.sum(i*ki[-nt:])-np.sum(i)*np.sum(ki[-nt:]))/(nt*np.sum(i**2)-np.sum(i)**2)
 m1=np.mean(kt);m2=np.mean((kt-m1)**2);pk=np.array([np.sum(kt==v) for v in np.unique(kt)])/W
 return np.array([ki[-1],ki[-1]/max(1.,np.max(kt)),kld(ki,ko),hs(kt),m1,np.mean((kt-m1)**3)/(m2**1.5) if m2>0 else 0.,gr/ga,sl,np.max(kt),np.std(kt)/max(m1,1.),np.mean(ko[-nb:]-ki[-nb:]),-np.sum(pk*np.log(pk+1e-12))])
@njit(fastmath=True)
def ftr(s):
 r=np.diff(s)/s[:-1];W,mr=len(s),np.mean(r);v2=np.mean((r-mr)**2);d=np.diff(s[-15:])
 return np.array([s[-1]/s[0]-1,np.std(r),np.mean((r-mr)**3)/(v2**1.5) if v2>0 else 0.,np.mean((r-mr)**4)/(v2**2) if v2>0 else 0.,100. if W<15 or np.mean(np.maximum(-d,0.))==0 else 100.-100./(1.+np.mean(np.maximum(d,0.))/np.mean(np.maximum(-d,0.))),(s[-1]-np.min(s))/max(np.max(s)-np.min(s),1e-8),s[-1]/(np.mean(s[-20:]) if W>=20 else s[-1])-1])
@njit(parallel=True,fastmath=True)
def bds(p,v=40,h=5):
 M=len(p)-v-h;X,D,R,P=np.zeros((M,19)),np.zeros(M),np.zeros(M),np.zeros(M)
 for i in prange(M):
  fv=fvg(p[i:i+v+1]);tr=ftr(p[i:i+v+1])
  for j in range(12):X[i,j]=fv[j]
  for j in range(7):X[i,12+j]=tr[j]
  R[i]=(p[i+v+h]-p[i+v])/p[i+v];D[i]=1. if R[i]>0 else 0.;P[i]=p[i+v+h]
 return X,D,R,P
class VGSS:
 def __init__(S,tk="AAPL",v=40,h=5,rt=0.7):
  d_train=yf.download(tk,start="2015-01-01",end="2025-12-31",progress=False)
  d_test=yf.download(tk,start="2026-01-01",progress=False)
  if isinstance(d_train.columns,pd.MultiIndex):
   p1=np.ascontiguousarray(d_train[('Close',tk)].values.ravel().astype(np.float64))
   p2=np.ascontiguousarray(d_test[('Close',tk)].values.ravel().astype(np.float64))
  else:
   p1=np.ascontiguousarray(d_train['Close'].values.ravel().astype(np.float64))
   p2=np.ascontiguousarray(d_test['Close'].values.ravel().astype(np.float64))
  S.p=np.concatenate([p1,p2]);S.tk=tk;S.X,S.y,S.R,S.P=bds(S.p,v,h);S.tr=int(len(S.y)*rt)
 def run(S):
  Xs=SS().fit_transform(S.X);t,nt,RE=S.tr,len(S.y)-S.tr,50
  bc=[CBC(verbose=0,iterations=200,random_seed=42),LC(n_estimators=200,verbose=-1,random_state=42),XC(n_estimators=200,verbosity=0,eval_metric='logloss',random_state=42)]
  br=[CBR(verbose=0,iterations=200,random_seed=42),LGR(n_estimators=200,verbose=-1,random_state=42),XR(n_estimators=200,verbosity=0,random_state=42)]
  t2=int(t*.7);print(f"⏳ Fase 1/3: Meta-features iniciales ({t2}→{t-t2} val)...")
  [m.fit(Xs[:t2],S.y[:t2]) for m in bc];[m.fit(Xs[:t2],S.P[:t2]) for m in br]
  Mc0=np.column_stack([m.predict_proba(Xs[t2:t])[:,1] for m in bc])
  Mr0=np.column_stack([m.predict(Xs[t2:t]) for m in br])
  mc,mr=LR(max_iter=2000),RD();mc.fit(Mc0,S.y[t2:t]);mr.fit(Mr0,S.P[t2:t])
  print(f"⏳ Fase 2/3: Base completo ({t} muestras)...")
  [m.fit(Xs[:t],S.y[:t]) for m in bc];[m.fit(Xs[:t],S.P[:t]) for m in br]
  bs0=t-t2;bsz=bs0+nt;Mcb,Mrb,ycb,prb=np.zeros((bsz,3)),np.zeros((bsz,3)),np.zeros(bsz),np.zeros(bsz)
  Mcb[:bs0],Mrb[:bs0],ycb[:bs0],prb[:bs0]=Mc0,Mr0,S.y[t2:t],S.P[t2:t];bs=bs0
  pd_a,pp_a=np.zeros(nt),np.zeros(nt)
  print(f"⏳ Fase 3/3: Walk-forward adaptivo ({nt} pasos)...")
  for i in range(nt):
   xi=Xs[t+i:t+i+1];ci=np.array([m.predict_proba(xi)[:,1][0] for m in bc]);ri=np.array([m.predict(xi)[0] for m in br])
   pd_a[i]=mc.predict(ci.reshape(1,-1))[0];pp_a[i]=mr.predict(ri.reshape(1,-1))[0]
   Mcb[bs],Mrb[bs],ycb[bs],prb[bs]=ci,ri,S.y[t+i],S.P[t+i];bs+=1
   sw=np.exp(np.linspace(-3,0,bs));mc.fit(Mcb[:bs],ycb[:bs],sample_weight=sw);mr.fit(Mrb[:bs],prb[:bs],sample_weight=sw)
   if(i+1)%RE==0:e=t+i+1;[m.fit(Xs[:e],S.y[:e]) for m in bc];[m.fit(Xs[:e],S.P[:e]) for m in br];print(f"  🔄 Paso {i+1}/{nt} | Acc:{AC(S.y[t:t+i+1],pd_a[:i+1]):.4f} R²:{RS(S.P[t:t+i+1],pp_a[:i+1]):.4f}")
  ye,pe=S.y[t:],S.P[t:];a,r2=AC(ye,pd_a),RS(pe,pp_a);mn=['CatBoost','LightGBM','XGBoost']
  print(f"\n{'='*60}\n✅ RESULTADOS\n{'='*60}\n  Dirección Acc: {a:.4f}\n  Precio R²:     {r2:.4f}\n{'='*60}")
  print(f"  Pesos CLF: {dict(zip(mn,mc.coef_[0].round(4)))}\n  Pesos REG: {dict(zip(mn,mr.coef_.round(4)))}")
  plt.style.use('default');plt.rcParams.update({'figure.facecolor':'w','axes.facecolor':'w','axes.grid':True,'grid.alpha':.25})
  fig,(a1,a2)=plt.subplots(2,1,figsize=(16,10),gridspec_kw={'hspace':.4});fig.patch.set_facecolor('w');n=len(ye);ok=(pd_a==ye)
  a1.plot(pe,c='#1565c0',lw=1.3,label='Real');a1.plot(pp_a,c='#c62828',lw=1.1,label='Predicho',alpha=.8)
  a1.fill_between(range(n),pe,pp_a,alpha=.08,color='#e65100')
  a1.set_title(f'{S.tk} Precio Cierre [Test] │ Stacking Adaptivo (CB+LGB+XGB)→Ridge │ R²={r2:.4f}',fontsize=13,fontweight='bold')
  a1.legend(fontsize=11);a1.set_ylabel('Precio ($)',fontsize=11)
  a2.fill_between(range(n),-.15,1.15,where=ok,color='#2e7d32',alpha=.1,label='Acierto')
  a2.fill_between(range(n),-.15,1.15,where=~ok,color='#c62828',alpha=.1,label='Error')
  a2.step(range(n),ye,c='#1565c0',lw=1.5,where='mid',label='Real');a2.step(range(n),pd_a,c='#c62828',lw=1.1,where='mid',label='Predicho',alpha=.7)
  a2.set_title(f'{S.tk} Dirección [Test] │ Stacking Adaptivo (CB+LGB+XGB)→LogReg │ Acc={a:.4f}',fontsize=13,fontweight='bold')
  a2.legend(fontsize=10,ncol=4,loc='upper center');a2.set_ylabel('0=Baja  1=Sube',fontsize=11);a2.set_ylim(-.2,1.2);a2.set_yticks([0,1])
  plt.xlabel('Muestras Test',fontsize=11);plt.savefig(f'{S.tk}_adaptive.png',dpi=150,bbox_inches='tight',facecolor='w');plt.show()
if __name__=="__main__":VGSS().run()

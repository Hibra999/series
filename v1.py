import numpy as np,pandas as pd,yfinance as yf,matplotlib.pyplot as plt;from numba import njit,prange;from sklearn.ensemble import GradientBoostingClassifier as GBC;from sklearn.preprocessing import StandardScaler as SS;from sklearn.metrics import accuracy_score as acc,confusion_matrix as cm,roc_curve as roc,precision_recall_curve as prc;import warnings;warnings.filterwarnings('ignore');import os,base64,io
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
 for k in kt:c[int(k)]+=1
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
 M=len(p)-v-h;X,D,R=np.zeros((M,19)),np.zeros(M),np.zeros(M)
 for i in prange(M):
  fv=fvg(p[i:i+v+1]);tr=ftr(p[i:i+v+1])
  for j in range(12):X[i,j]=fv[j]
  for j in range(7):X[i,12+j]=tr[j]
  R[i]=(p[i+v+h]-p[i+v])/p[i+v];D[i]=1. if R[i]>0 else 0.
 return X,D,R
class VGFS:
 def __init__(S,tk="AAPL",v=40,h=5,rt=0.7):
  d=yf.download(tk,start="2020-01-01",end="2025-01-01",progress=False)
  if isinstance(d.columns, pd.MultiIndex):
      S.p,S.tk=np.ascontiguousarray(d[('Close',tk)].values.ravel().astype(np.float64)),tk
  else:
      S.p,S.tk=np.ascontiguousarray(d['Close'].values.ravel().astype(np.float64)),tk
  S.X,S.y,S.R=bds(S.p,v,h)
  S.tr=int(len(S.y)*rt)
  S.rs,S.es,S.imgs={},{},{}
 def run(S):
  Xs,tr=SS().fit_transform(S.X),S.tr
  for n,idx in [('VG',list(range(12))),('TR',list(range(12,19))),('ALL',list(range(19)))]:
   m=GBC(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42).fit(Xs[:tr,idx],S.y[:tr]);yp,yprob=m.predict(Xs[tr:,idx]),m.predict_proba(Xs[tr:,idx])[:,1];S.rs[n]={'yp':yp,'yprob':yprob,'acc':acc(S.y[tr:],yp),'cm':cm(S.y[tr:],yp)}
  bn=max(S.rs,key=lambda k:S.rs[k]['acc']);b=S.rs[bn];R=S.R[tr:];C=lambda y:np.cumprod(1+y)-1
  for n,r in S.rs.items():S.es[n]=C(np.where(r['yp']==1,R,0))
  S.es['B&H']=C(R);plt.style.use('dark_background');F=plt.figure;S.p1(b);S.p2(b,R);S.p3();S.p4(b,R);S.p5();S.gen_html(b,R,bn);print(f"✅ Mejor: {bn} (Acc: {b['acc']:.2%})")
 def img2b64(S,f):buf=io.BytesIO();f.savefig(buf,format='png',bbox_inches='tight');buf.seek(0);return base64.b64encode(buf.read()).decode('utf-8')
 def p1(S,b):f=plt.figure(figsize=(15,10));ax=f.subplots(2,2);ax[0,0].plot(S.p);ax[0,0].axvline(S.tr,c='y');ax[0,0].set_title('Precio AAPL');ax[0,1].plot(S.es['VG'],label='VG');ax[0,1].plot(S.es['TR'],label='TR');ax[0,1].plot(S.es['ALL'],label='ALL');ax[0,1].plot(S.es['B&H'],label='B&H',ls='--');ax[0,1].legend();ax[0,1].set_title('Retornos Acumulados');ax[1,0].plot(S.X[:,3]);ax[1,0].set_title('Exponente Hurst');ax[1,1].imshow(b['cm'],cmap='Blues');ax[1,1].set_title('Matriz Confusión');S.imgs['p1']=S.img2b64(f);plt.close(f)
 def p2(S,b,R):f=plt.figure(figsize=(15,10));ax=f.subplots(2,2);y=S.y[S.tr:];fpr,tpr,_=roc(y,b['yprob']);ax[0,0].bar(range(len(R)),R,color=['g' if p==a else 'r' for p,a in zip(b['yp'],y)]);ax[0,0].set_title('Retornos por Predicción');ax[0,1].plot(fpr,tpr);ax[0,1].set_title('Curva ROC');pr,rc,_=prc(y,b['yprob']);ax[1,0].plot(rc,pr);ax[1,0].set_title('Curva P-R');ax[1,1].plot(S.es['ALL'],c='g',label='VG+TR');ax[1,1].plot(S.es['B&H'],c='w',ls='--',label='B&H');ax[1,1].legend();ax[1,1].set_title('Comparativa Estrategias');S.imgs['p2']=S.img2b64(f);plt.close(f)
 def p3(S):f=plt.figure(figsize=(15,5));ax=f.subplots(1,3);_,_,kt=vg(S.p);u,c=np.unique(kt.astype(np.int32),return_counts=True);ax[0].loglog(u,c/len(S.p),'.');ax[0].set_title('Distribución de Grados');ax[1].plot(kt);ax[1].set_title('Secuencia de Grados');ax[2].scatter(S.X[:,3],S.R,c=S.y,alpha=0.5);ax[2].set_title('Hurst vs Retornos');S.imgs['p3']=S.img2b64(f);plt.close(f)
 def p4(S,b,R):f=plt.figure(figsize=(15,10));ax=f.subplots(4,1);n=min(120,len(R));idx=slice(-n,None);y,yp=S.y[S.tr:][idx],b['yp'][idx];ax[0].plot(S.p[-n:]);ax[0].set_title('Precio (Últimos 120)');ax[1].bar(range(n),b['yprob'][idx]);ax[1].set_title('Probabilidad de Subida');ax[2].bar(range(n),R[idx]);ax[2].set_title('Retornos');ax[3].plot(np.cumsum(yp==y)/(np.arange(n)+1)*100);ax[3].set_title('Precisión Acumulada (%)');S.imgs['p4']=S.img2b64(f);plt.close(f)
 def p5(S):f=plt.figure(figsize=(10,5));ax=f.subplots(1,2);ax[0].barh(range(19),np.corrcoef(S.X.T,S.R)[-1,:-1]);ax[0].set_title('Correlación con Retorno');ax[1].imshow(np.corrcoef(S.X.T),cmap='RdBu_r');ax[1].set_title('Heatmap Correlaciones');S.imgs['p5']=S.img2b64(f);plt.close(f)
 def gen_html(S,b,R,bn):
  pngs=[f for f in os.listdir('.') if f.endswith('.png')]
  for p in pngs:os.remove(p);print(f"🗑️ Eliminado: {p}")
  feat_names=['In-degree','Norm In-degree','KLD','Hurst','Mean Degree','Skewness','Growth Ratio','Slope','Max Degree','CV','Out-In Diff','Entropy','Total Return','Volatility','Skew','Kurtosis','Drawdown','Rel Position','Momentum']
  html=f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Reporte VGFS - {S.tk}</title><style>body{{font-family:Arial,sans-serif;background:#1a1a2e;color:#eee;padding:20px}}h1,h2{{color:#00d9ff}}table{{border-collapse:collapse;width:100%;margin:20px 0;background:#16213e}}th,td{{border:1px solid #444;padding:10px;text-align:center}}th{{background:#0f3460;color:#00d9ff}}.pos{{color:#4caf50}}.neg{{color:#f44336}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}}.card{{background:#16213e;padding:15px;border-radius:8px}}img{{width:100%;border-radius:8px}}</style></head><body>
<h1>📊 Reporte Visibility Graph Forecasting System - {S.tk}</h1>
<h2>📈 Resumen de Modelos</h2>
<table><tr><th>Modelo</th><th>Accuracy</th><th>Verdaderos Positivos</th><th>Falsos Positivos</th><th>Falsos Negativos</th><th>Verdaderos Negativos</th></tr>"""
  for n in ['VG','TR','ALL']:
   r=S.rs[n];tn,fp,fn,tp=r['cm'].ravel();html+=f"<tr><td>{n}</td><td>{r['acc']:.2%}</td><td class='pos'>{tp}</td><td class='neg'>{fp}</td><td class='neg'>{fn}</td><td class='pos'>{tn}</td></tr>"
  html+=f"</table><p><strong>✅ Mejor Modelo:</strong> {bn} con accuracy {b['acc']:.2%}</p>"
  html+="<h2>📉 Estrategias de Retorno</h2><table><tr><th>Estrategia</th><th>Retorno Final</th><th>Máximo</th><th>Mínimo</th></tr>"
  for n,e in S.es.items():ret=e[-1];mx,mn=e.max(),e.min();c='#4caf50' if ret>=0 else '#f44336';html+=f"<tr><td>{n}</td><td style='color:{c}'>{ret:.2%}</td><td style='color:#4caf50'>{mx:.2%}</td><td style='color:#f44336'>{mn:.2%}</td></tr>"
  html+=f"</table><h2>📊 Dashboard Principal</h2><div class='grid'><div class='card'><img src='data:image/png;base64,{S.imgs['p1']}'></div><div class='card'><img src='data:image/png;base64,{S.imgs['p2']}'></div></div>"
  html+=f"<h2>🔍 Análisis VG</h2><div class='grid'><div class='card'><img src='data:image/png;base64,{S.imgs['p3']}'></div><div class='card'><img src='data:image/png;base64,{S.imgs['p4']}'></div></div>"
  html+=f"<h2>📐 Análisis de Features</h2><div class='grid'><div class='card'><img src='data:image/png;base64,{S.imgs['p5']}'></div></div>"
  html+="<h2>📋 Features VG (Top 12)</h2><table><tr><th>Feature</th><th>Valor Actual</th><th>Media Histórica</th></tr>"
  vnames=['In-degree','Norm In-degree','KLD (Irreversibilidad)','Hurst','Mean Degree','Skewness','Growth Ratio','Slope','Max Degree','CV','Out-In Diff','Entropy']
  for i,vn in enumerate(vnames):cur=S.X[-1,i];avg=np.mean(S.X[:,i]);html+=f"<tr><td>{vn}</td><td>{cur:.4f}</td><td>{avg:.4f}</td></tr>"
  html+=f"</table><h2>📋 Features Técnicos (Top 7)</h2><table><tr><th>Feature</th><th>Valor Actual</th><th>Media Histórica</th></tr>"
  tnames=['Retorno Total','Volatilidad','Skew','Kurtosis','Drawdown Ratio','Posición Relativa','Momentum']
  for i,tn in enumerate(tnames):cur=S.X[-1,12+i];avg=np.mean(S.X[:,12+i]);html+=f"<tr><td>{tn}</td><td>{cur:.4f}</td><td>{avg:.4f}</td></tr>"
  html+=f"</table><p style='text-align:center;color:#666;margin-top:40px'>Generado por VGFS | Visibility Graph Forecasting System</p></body></html>"""
  S.imgs={};fn=f"{S.tk}_reporte.html";open(fn,'w',encoding='utf-8').write(html);print(f"✅ HTML generado: {fn}")
if __name__=="__main__":VGFS().run()

import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; from numba import njit, prange; from scipy import stats; import time; import warnings; warnings.filterwarnings('ignore')
try: import yfinance as yf; HAS_YF=True
except: HAS_YF=False; print("⚠ pip install yfinance")

@njit(parallel=True, fastmath={'contract', 'reassoc'})
def _build_vg_core(serie, k_out, k_in, N):
    for i in prange(N-1):
        pm=-np.inf
        for j in range(i+1,N):
            p=(serie[j]-serie[i])/(j-i)
            if p>pm: k_out[i]+=1; k_in[j]+=1; pm=p

def descargar_datos(t="AAPL",i="2020-01-01",f="2025-01-01"):
    if HAS_YF:
        df=yf.download(t,start=i,end=f,progress=False)
        return df['Close'].values.flatten(),df.index
    np.random.seed(42); N=1260; ret=np.random.normal(0.0005,0.018,N)
    return 75*np.exp(np.cumsum(ret)),pd.date_range(i,periods=N,freq='B')

def construir_visibility_graph(s,g=True):
    N=len(s); ar=[] if g else None; ko=np.zeros(N,np.int64); ki=np.zeros(N,np.int64)
    _build_vg_core(s,ko,ki,N)
    if g:
        idx=np.where(ko>0)[0]
        for i in idx:
            pm=-np.inf
            for j in range(i+1,N):
                p=(s[j]-s[i])/(j-i)
                if p>pm: ar.append((i,j)); pm=p
    return (ar,ki,ko,ki+ko) if g else (ki,ko,ki+ko)

def distribucion_grado(g):
    u,c=np.unique(g,return_counts=True); return u,c/len(g)

def calcular_kld(ki,ko,g=1):
    N=len(ki); eps=g/N
    ui,ci=np.unique(ki,return_counts=True); uo,co=np.unique(ko,return_counts=True)
    uk=np.union1d(ui,uo)
    pi=np.array([ci[np.searchsorted(ui,k)] if k in ui else 0 for k in uk])/N+eps
    po=np.array([co[np.searchsorted(uo,k)] if k in uo else 0 for k in uk])/N+eps
    pi/=pi.sum(); po/=po.sum()
    return np.sum(pi*np.log(pi/po))

def ajustar_ley_potencia(k,p,kmin=2):
    m=(k>=kmin)&(p>0)
    if m.sum()<3: return np.nan,np.nan,np.nan
    lk=np.log(k[m].astype(float)); lp=np.log(p[m])
    r,i,rv,_,_=stats.linregress(lk,lp)
    return -r,(3-(-r))/2,rv**2

@njit(fastmath=True)
def vg_fuerza_bruta(s):
    N=len(s); ar=[]
    for i in range(N):
        for j in range(i+1,N):
            v=True
            for k in range(i+1,j):
                if s[k]>=s[i]+(s[j]-s[i])*(k-i)/(j-i): v=False; break
            if v: ar.append((i,j))
    return ar

if __name__=="__main__":
    print("="*65+"\n  VISIBILITY GRAPH ANALYSIS — APPLE (AAPL) 2020-2025\n"+"="*65)
    p,f=descargar_datos("AAPL","2020-01-01","2025-01-01"); N=len(p)
    print(f"\n📊 Datos: {N} días | {f[0].strftime('%Y-%m-%d')}→{f[-1].strftime('%Y-%m-%d')} | ${p.min():.2f}-${p.max():.2f}")
    print("\n🔍 Verificando (20 datos)..."); sb=p[:20]
    ab=vg_fuerza_bruta(sb); ar,_,_,_=construir_visibility_graph(sb)
    print("   ✅ OK" if set(ab)==set(ar) else "   ❌ ERROR")
    print(f"\n🔨 Construyendo VG ({N} nodos)..."); t0=time.time()
    ar,ki,ko,kt=construir_visibility_graph(p); tv=time.time()-t0
    print(f"   ⏱ {tv:.1f}s | {len(ar)} aristas | grado: {kt.mean():.2f} (max:{kt.max()})")
    kiv,piv=distribucion_grado(ki); kov,pov=distribucion_grado(ko); ktv,ptv=distribucion_grado(kt)
    print("\n"+"="*65+"\n  IRREVERSIBILIDAD TEMPORAL\n"+"="*65)
    kld=calcular_kld(ki,ko); print(f"\n   KLD = {kld:.6f}")
    ka=[]; print("\n   Control (shuffle)...")
    for s in range(3):
        np.random.seed(s+100); sh=p.copy(); np.random.shuffle(sh)
        kia,koa,_=construir_visibility_graph(sh,False); ka.append(calcular_kld(kia,koa))
        print(f"      #{s+1}: {ka[-1]:.6f}")
    km=np.mean(ka); print(f"\n   Original: {kld:.6f} | Shuffle: {km:.6f} | Ratio: {kld/km:.1f}x")
    print("\n   ✅ IRREVERSIBLE" if kld>3*km else "\n   ⚠️ No concluyente")
    print("\n"+"="*65+"\n  ANÁLISIS FRACTAL\n"+"="*65)
    r,H,r2=ajustar_ley_potencia(ktv,ptv)
    print(f"\n   P(k)~k^(-{r:.4f}) | R²={r2:.4f} | H={H:.4f}")
    print(f"   📈 PERSISTENTE" if H>0.5 else "   📉 ANTI-PERSISTENTE" if H<0.5 else "   ↔️ Aleatorio")
    fig,ax=plt.subplots(2,3,figsize=(20,12)); fig.suptitle('Apple VG 2020-2025',fontsize=16,fontweight='bold')
    ax[0,0].plot(f,p,'steelblue',lw=0.8); ax[0,0].fill_between(f,p,alpha=0.1); ax[0,0].set_title('Precio'); ax[0,0].grid(True,alpha=0.3)
    ax[0,1].scatter(kiv,piv,c='royalblue',s=50,alpha=0.7,label='P_in'); ax[0,1].scatter(kov,pov,c='crimson',s=50,alpha=0.7,label='P_out')
    ax[0,1].set_xscale('log'); ax[0,1].set_yscale('log'); ax[0,1].set_title(f'KLD={kld:.4f}'); ax[0,1].legend(); ax[0,1].grid(True,alpha=0.3)
    ax[0,2].scatter(ktv,ptv,c='forestgreen',s=50,alpha=0.7); mx=(ktv>=2)&(ptv>0)
    if mx.any():
        ka_=ktv[mx].astype(float); pp=np.exp(np.log(ptv[mx][0])+(-r)*(np.log(ka_)-np.log(ka_[0])))
        ax[0,2].plot(ka_,pp,'r--',lw=2.5,label=f'k^{-r:.2f} (H={H:.2f})')
    ax[0,2].set_xscale('log'); ax[0,2].set_yscale('log'); ax[0,2].set_title('Ley de potencia'); ax[0,2].legend(); ax[0,2].grid(True,alpha=0.3)
    ax[1,0].plot(range(N),ko,'r-',lw=0.4,alpha=0.6,label='k_out'); ax[1,0].plot(range(N),ki,'b-',lw=0.4,alpha=0.6,label='k_in')
    ax[1,0].set_title('Secuencias'); ax[1,0].legend(); ax[1,0].grid(True,alpha=0.3)
    kb=min(40,max(kiv.max(),kov.max())+1); rk=np.arange(kb+1)
    pid=dict(zip(kiv,piv)); pod=dict(zip(kov,pov))
    ax[1,1].bar(rk-0.175,[pid.get(k,0) for k in rk],0.35,alpha=0.7,color='royalblue',label='P_in')
    ax[1,1].bar(rk+0.175,[pod.get(k,0) for k in rk],0.35,alpha=0.7,color='crimson',label='P_out')
    ax[1,1].set_title('Asimetría'); ax[1,1].legend(); ax[1,1].grid(True,alpha=0.3,axis='y')
    nv=min(50,N); pv=p[:nv]; ax[1,2].bar(range(nv),pv,color='steelblue',alpha=0.6)
    for i,j in [(i,j) for i,j in ar if i<nv and j<nv]: ax[1,2].plot([i,j],[pv[i],pv[j]],'r-',alpha=0.12,lw=0.6)
    ax[1,2].set_title(f'VG ({nv} nodos)'); ax[1,2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig('apple_visibility_graph.png',dpi=150,bbox_inches='tight'); plt.show()
    print("\n"+"="*65+"\n  RESUMEN\n"+"="*65+f"\n    Aristas: {len(ar)} | Grado: {kt.mean():.2f} | KLD: {kld:.6f} | H: {H:.4f}\n")

import numpy as np,pandas as pd,yfinance as yf,matplotlib.pyplot as plt,torch,torch.nn as nn,torch.nn.functional as F,torch.fft,math,warnings
from numba import njit,prange
from scipy.special import logit,expit
from scipy.interpolate import PchipInterpolator
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler as SS,QuantileTransformer,PowerTransformer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score as AC,r2_score as RS
from catboost import CatBoostClassifier as CBC,CatBoostRegressor as CBR
from lightgbm import LGBMClassifier as LC,LGBMRegressor as LGR
from xgboost import XGBClassifier as XC,XGBRegressor as XR
warnings.filterwarnings('ignore')

@njit(fastmath=True,cache=True)
def vg(s):
    N=len(s);ki=np.zeros(N,dtype=np.float64);ko=np.zeros(N,dtype=np.float64)
    for i in range(N-1):
        m=-1e15
        for j in range(i+1,N):
            d=float(j-i)
            if d>0:
                p=(s[j]-s[i])/d
                if p>m:
                    ko[i]+=1.0;ki[j]+=1.0;m=p
    return ki,ko,ki+ko

@njit(fastmath=True,cache=True)
def kld(ki,ko):
    N=len(ki);mx=int(max(np.max(ki),np.max(ko)))+1;ci=np.zeros(mx,dtype=np.float64);co=np.zeros(mx,dtype=np.float64)
    for i in range(N):
        ki_idx=int(ki[i]);ko_idx=int(ko[i])
        if 0<=ki_idx<mx:ci[ki_idx]+=1.0
        if 0<=ko_idx<mx:co[ko_idx]+=1.0
    pi=(ci/N)+(1.0/N);po=(co/N)+(1.0/N);pi=pi/np.sum(pi);po=po/np.sum(po);r=0.0
    for i in range(len(pi)):
        if pi[i]>0 and po[i]>0:r+=pi[i]*np.log(pi[i]/po[i])
    return r

@njit(fastmath=True,cache=True)
def hs(kt):
    c=np.zeros(int(np.max(kt))+1,dtype=np.float64)
    for v in kt:
        vi=int(v)
        if 0<=vi<len(c):c[vi]+=1.0
    k=np.nonzero(c)[0];p=c[k]/len(kt);m=(k>=2)&(p>0)
    if np.sum(m)<3:return 0.5
    x=np.log(k[m].astype(np.float64));y=np.log(p[m]);n=len(x);sx=np.sum(x)
    h=(3+(n*np.sum(x*y)-sx*np.sum(y))/(n*np.sum(x*x)-sx**2))/2
    return max(0.01,min(0.99,h))

@njit(fastmath=True,cache=True)
def fvg(s):
    ki,ko,kt=vg(s);W=len(s);nb=max(3,W//10);nt=min(10,W//3);gr=np.mean(kt[-nb:]);ga=max(1.0,np.mean(kt[:nb]));sl=0.0
    if nt>=3:
        ii=np.arange(nt,dtype=np.float64);dn=nt*np.sum(ii**2)-np.sum(ii)**2
        if dn!=0:sl=(nt*np.sum(ii*ki[-nt:])-np.sum(ii)*np.sum(ki[-nt:]))/dn
    m1=np.mean(kt);m2=np.mean((kt-m1)**2);uv=np.unique(kt);pk=np.array([np.sum(kt==v) for v in uv])/W
    skew=np.mean((kt-m1)**3)/(m2**1.5) if m2>0 else 0.0
    entropy=0.0
    for p in pk:
        if p>0:entropy-=p*np.log(p+1e-12)
    return np.array([ki[-1],ki[-1]/max(1.0,np.max(kt)),kld(ki,ko),hs(kt),m1,skew,gr/ga,sl,np.max(kt),np.std(kt)/max(m1,1.0),np.mean(ko[-nb:]-ki[-nb:]),entropy])

@njit(fastmath=True,cache=True)
def ftr(s):
    r=np.diff(s)/s[:-1];W=len(s);mr=np.mean(r);v2=np.mean((r-mr)**2)
    d=np.diff(s[-15:]) if W>=15 else np.array([0.0]);md=np.mean(np.maximum(d,0.0));mu=np.mean(np.maximum(-d,0.0))
    dd=100.0 if W<15 or mu==0 else 100.0-100.0/(1.0+md/mu);pr=max(np.max(s)-np.min(s),1e-8);ma=np.mean(s[-20:]) if W>=20 else s[-1]
    return np.array([s[-1]/s[0]-1,np.std(r),np.mean((r-mr)**3)/(v2**1.5) if v2>0 else 0.0,np.mean((r-mr)**4)/(v2**2) if v2>0 else 0.0,dd,(s[-1]-np.min(s))/pr,s[-1]/ma-1])

@njit(fastmath=True,cache=True)
def bds_multi(p,v=40,horizons=np.array([1,3,5,10])):
    hmax=int(np.max(horizons));M=len(p)-v-hmax;nh=len(horizons)
    X=np.zeros((M,19),dtype=np.float64);D=np.zeros((M,nh),dtype=np.float64);R=np.zeros((M,nh),dtype=np.float64);P=np.zeros((M,nh),dtype=np.float64)
    for i in range(M):
        fv=fvg(p[i:i+v+1]);tr=ftr(p[i:i+v+1])
        for j in range(12):X[i,j]=fv[j]
        for j in range(7):X[i,12+j]=tr[j]
        for hi in range(nh):
            h=int(horizons[hi]);R[i,hi]=(p[i+v+h]-p[i+v])/p[i+v];D[i,hi]=1.0 if R[i,hi]>0 else 0.0;P[i,hi]=p[i+v+h]
    return X,D,R,P

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):
        return self.pe[:,:x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self,c_in,d_model):
        super().__init__()
        self.conv=nn.Conv1d(c_in,d_model,3,padding=1,padding_mode='circular',bias=False)
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_in',nonlinearity='leaky_relu')
    def forward(self,x):
        return self.conv(x.permute(0,2,1)).transpose(1,2)

class DataEmbedding(nn.Module):
    def __init__(self,c_in,d_model,dropout=0.1):
        super().__init__()
        self.value_embedding=TokenEmbedding(c_in,d_model)
        self.position_embedding=PositionalEmbedding(d_model)
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,x):
        return self.dropout(self.value_embedding(x)+self.position_embedding(x))

class Inception_Block_V1(nn.Module):
    def __init__(self,in_ch,out_ch,num_kernels=6):
        super().__init__()
        self.kernels=nn.ModuleList([nn.Conv2d(in_ch,out_ch,kernel_size=2*i+1,padding=i) for i in range(num_kernels)])
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:nn.init.constant_(m.bias,0)
    def forward(self,x):
        return torch.stack([k(x) for k in self.kernels],dim=-1).mean(-1)

class TimesBlock(nn.Module):
    def __init__(self,seq_len,pred_len,d_model,d_ff,top_k,num_kernels):
        super().__init__()
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.k=top_k
        self.conv=nn.Sequential(
            Inception_Block_V1(d_model,d_ff,num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff,d_model,num_kernels)
        )
    def forward(self,x):
        B,T,N=x.size()
        # Protección contra secuencias muy cortas
        if T < 4:
            return x
        
        xf=torch.fft.rfft(x,dim=1)
        fl=abs(xf).mean(0).mean(-1)
        fl[0]=0
        
        # Limitar k al número de frecuencias disponibles
        actual_k = min(self.k, len(fl)-1, T//2)
        if actual_k < 1:
            return x
            
        _,tl=torch.topk(fl,actual_k)
        tl=tl.detach().cpu().numpy()
        
        # Evitar división por cero
        tl = np.maximum(tl, 1)
        periods = T // tl
        periods = np.maximum(periods, 1)
        
        pw=abs(xf).mean(-1)[:,tl]
        res=[]
        
        for i in range(actual_k):
            p=int(periods[i])
            if p < 1: p = 1
            
            length = ((T // p) + 1) * p if T % p != 0 else T
            if length > T:
                pad = torch.zeros([B, length - T, N], device=x.device)
                out = torch.cat([x, pad], dim=1)
            else:
                out = x
                length = T
            
            n_periods = length // p
            if n_periods < 1: n_periods = 1
            
            out = out[:, :n_periods*p, :].reshape(B, n_periods, p, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        
        if len(res) == 0:
            return x
            
        res = torch.stack(res, dim=-1)
        pw = F.softmax(pw, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        return torch.sum(res * pw, -1) + x

class TimesNetMeta(nn.Module):
    def __init__(self,enc_in,seq_len=20,pred_len=1,d_model=32,d_ff=32,e_layers=2,top_k=3,num_kernels=4,dropout=0.1,c_out=1):
        super().__init__()
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.enc_embedding=DataEmbedding(enc_in,d_model,dropout)
        self.model=nn.ModuleList([TimesBlock(seq_len,pred_len,d_model,d_ff,top_k,num_kernels) for _ in range(e_layers)])
        self.layer_norm=nn.LayerNorm(d_model)
        self.projection=nn.Linear(d_model,c_out,bias=True)
        self.layer=e_layers
        
    def forward(self,x):
        # x: (batch, seq_len, features)
        B, T, _ = x.size()
        
        means=x.mean(1,keepdim=True).detach()
        x=x-means
        stdev=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5)
        x=x/stdev
        
        enc_out=self.enc_embedding(x)  # (B, T, d_model)
        
        for i in range(self.layer):
            enc_out=self.layer_norm(self.model[i](enc_out))
        
        # Solo tomar la última posición para predicción
        dec_out=self.projection(enc_out[:, -1, :])  # (B, c_out)
        
        # Denormalizar
        dec_out=dec_out*stdev[:, 0, 0:1]+means[:, 0, 0:1]
        
        return dec_out  # (B, 1)

class TimesNetMetaWrapper:
    def __init__(self,n_features,seq_len=20,lr=1e-3,epochs=50,d_model=32,d_ff=32,e_layers=2,top_k=3,num_kernels=4,dropout=0.1):
        self.seq_len=seq_len
        self.lr=lr
        self.epochs=epochs
        self.n_features=n_features
        self.net=TimesNetMeta(
            enc_in=n_features,
            seq_len=seq_len,
            pred_len=1,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            top_k=top_k,
            num_kernels=num_kernels,
            dropout=dropout,
            c_out=1
        )
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=lr,weight_decay=1e-4)
        self.criterion=nn.MSELoss(reduction='none')
        self.scaler_x=None
        self.scaler_y=None
        
    def _build_sequences(self,X,y):
        n=len(X)
        seqs=[]
        targets=[]
        for i in range(self.seq_len,n):
            seqs.append(X[i-self.seq_len:i])
            targets.append(y[i])
        return np.array(seqs),np.array(targets)
    
    def fit(self,X,y,sample_weight=None):
        if len(X)<self.seq_len+2:
            return self
        
        from sklearn.preprocessing import StandardScaler
        self.scaler_x=StandardScaler().fit(X)
        self.scaler_y=StandardScaler().fit(y.reshape(-1,1))
        
        Xn=self.scaler_x.transform(X)
        yn=self.scaler_y.transform(y.reshape(-1,1)).ravel()
        
        Xs,ys=self._build_sequences(Xn,yn)  # CORREGIDO: underscore
        
        if len(Xs)<2:
            return self
        
        Xt=torch.tensor(Xs,dtype=torch.float32)
        yt=torch.tensor(ys,dtype=torch.float32)
        
        # Ajustar sample_weight para las secuencias
        if sample_weight is None:
            sw=torch.ones(len(yt))
        else:
            sw=torch.tensor(sample_weight[self.seq_len:len(sample_weight)],dtype=torch.float32)
            if len(sw) != len(yt):
                sw = torch.ones(len(yt))
        
        self.net.train()
        ds=torch.utils.data.TensorDataset(Xt,yt,sw)
        dl=torch.utils.data.DataLoader(ds,batch_size=min(64,len(ds)),shuffle=True)
        
        for _ in range(self.epochs):
            for xb,yb,wb in dl:
                self.optimizer.zero_grad()
                pred=self.net(xb).squeeze(-1)  # (batch,)
                loss=(self.criterion(pred,yb)*wb).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0)
                self.optimizer.step()
        return self
    
    def predict(self,X):
        if self.scaler_x is None:
            return np.zeros(len(X))
        
        Xn=self.scaler_x.transform(X)
        n=len(Xn)
        preds=[]
        
        self.net.eval()
        with torch.no_grad():
            for i in range(n):
                if i<self.seq_len:
                    seq=np.zeros((self.seq_len,self.n_features))
                    valid=min(i+1,self.seq_len)
                    seq[-valid:]=Xn[max(0,i+1-self.seq_len):i+1]
                else:
                    seq=Xn[i-self.seq_len:i]
                
                xt=torch.tensor(seq,dtype=torch.float32).unsqueeze(0)
                p=self.net(xt).squeeze().item()
                preds.append(p)
        
        return self.scaler_y.inverse_transform(np.array(preds).reshape(-1,1)).ravel()

class Transf_PIT_Logit:
    def fit(self,y):
        self.qt=QuantileTransformer(output_distribution='uniform',n_quantiles=min(100,len(y))).fit(y.reshape(-1,1))
    def transform(self,y):
        return logit(np.clip(self.qt.transform(y.reshape(-1,1)).ravel(),1e-6,1-1e-6))
    def inverse_transform(self,z):
        return self.qt.inverse_transform(expit(z).reshape(-1,1)).ravel()

class Transf_PyTorch_Flow:
    def __init__(self):
        self.w=nn.Parameter(torch.tensor(1.0))
        self.b=nn.Parameter(torch.tensor(0.0))
        self.opt=torch.optim.Adam([self.w,self.b],lr=0.1)
    def fit(self,y):
        yt=torch.tensor(y,dtype=torch.float32)
        for _ in range(100):
            self.opt.zero_grad()
            z=yt*self.w+self.b
            loss=torch.mean(0.5*z**2)-torch.log(torch.abs(self.w)+1e-6)
            loss.backward()
            self.opt.step()
    def transform(self,y):
        return(torch.tensor(y,dtype=torch.float32)*self.w+self.b).detach().numpy()
    def inverse_transform(self,z):
        return((torch.tensor(z,dtype=torch.float32)-self.b)/self.w).detach().numpy()

class Transf_Spline:
    def fit(self,y):
        self.q=np.linspace(0,1,15)
        self.y_q=np.quantile(y,self.q)
        self.z_q=QuantileTransformer(output_distribution='normal').fit_transform(self.y_q.reshape(-1,1)).ravel()
        self.z_q[0],self.z_q[-1]=-3.5,3.5
        self.fwd=PchipInterpolator(self.y_q,self.z_q)
        self.inv=PchipInterpolator(self.z_q,self.y_q)
    def transform(self,y):
        return self.fwd(np.clip(y,self.y_q[0],self.y_q[-1]))
    def inverse_transform(self,z):
        return self.inv(np.clip(z,self.z_q[0],self.z_q[-1]))

class Transf_INN_Power:
    def fit(self,y):
        self.pt=PowerTransformer(method='yeo-johnson',standardize=True).fit(y.reshape(-1,1))
    def transform(self,y):
        return self.pt.transform(y.reshape(-1,1)).ravel()
    def inverse_transform(self,z):
        return self.pt.inverse_transform(z.reshape(-1,1)).ravel()

class Transf_Sktime_Pipeline:
    def fit(self,y):
        self.ss=SS().fit(np.log(y+1e-8).reshape(-1,1))
    def transform(self,y):
        return self.ss.transform(np.log(y+1e-8).reshape(-1,1)).ravel()
    def inverse_transform(self,z):
        return np.exp(self.ss.inverse_transform(z.reshape(-1,1)).ravel())

class Transf_Omni_Custom:
    def fit(self,y):
        self.qt=QuantileTransformer(output_distribution='normal',n_quantiles=min(150,len(y))).fit(y.reshape(-1,1))
    def transform(self,y):
        return np.sinh(self.qt.transform(y.reshape(-1,1)).ravel())
    def inverse_transform(self,z):
        return self.qt.inverse_transform(np.arcsinh(z).reshape(-1,1)).ravel()

class VGSS:
    def __init__(self,tk="AAPL",v=40,horizons=[1,3,5,10],rt=0.7,err_window=10,err_thresh=1.5):
        print(f"📥 Descargando datos para {tk}...")
        d_train=yf.download(tk,start="2015-01-01",end="2023-12-31",progress=False)
        d_test=yf.download(tk,start="2024-01-01",progress=False)
        
        if isinstance(d_train.columns,pd.MultiIndex):
            p1=np.ascontiguousarray(d_train[('Close',tk)].values.ravel().astype(np.float64))
            p2=np.ascontiguousarray(d_test[('Close',tk)].values.ravel().astype(np.float64))
        else:
            p1=np.ascontiguousarray(d_train['Close'].values.ravel().astype(np.float64))
            p2=np.ascontiguousarray(d_test['Close'].values.ravel().astype(np.float64))
        
        self.p=np.concatenate([p1,p2])
        self.tk=tk
        self.horizons=np.array(horizons,dtype=np.int64)
        self.h_primary=5
        self.h_primary_idx=list(horizons).index(5) if 5 in horizons else 2
        self.err_window=err_window
        self.err_thresh=err_thresh
        
        print("⚙️ Generando features multi-horizon...")
        self.X,self.D,self.R,self.P=bds_multi(self.p,v,self.horizons)
        self.tr=int(len(self.D)*rt)
        self.stride=self.h_primary
        print(f"✅ {len(self.D)} muestras | train={self.tr} | stride={self.stride} | horizons={horizons}")

    def run_all(self):
        transformaciones={
            "PIT_Aprendida_Logit":Transf_PIT_Logit(),
            "Normalizing_Flow_PyTorch":Transf_PyTorch_Flow(),
            "Rational_Spline_PCHIP":Transf_Spline(),
            "INN_Yeo_Johnson":Transf_INN_Power(),
            "Sktime_Pipeline_LogSS":Transf_Sktime_Pipeline(),
            "Omni_Custom_SinhArcsinh":Transf_Omni_Custom()
        }
        resultados=[]
        for name,transformer in transformaciones.items():
            print(f"\n{'='*70}\n🚀 Evaluando: {name}\n{'='*70}")
            try:
                a,r2=self.run_single_transform(name,transformer)
                resultados.append((name,a,r2))
            except Exception as e:
                print(f"❌ Error en {name}: {e}")
                resultados.append((name,0.0,0.0))
        self.generar_html(resultados)

    def run_single_transform(self,t_name,transformer):
        Xs=SS().fit_transform(self.X)
        t=self.tr
        nh=len(self.horizons)
        hi=self.h_primary_idx
        stride=self.stride
        
        y_dir=self.D[:,hi]
        P_target=self.P[:,hi]
        
        bc=[
            CBC(verbose=0,iterations=150,random_seed=42,thread_count=-1),
            LC(n_estimators=150,verbose=-1,random_state=42,n_jobs=-1),
            XC(n_estimators=150,verbosity=0,eval_metric='logloss',random_state=42,n_jobs=-1)
        ]
        br_all=[[
            CBR(verbose=0,iterations=150,random_seed=42,thread_count=-1),
            LGR(n_estimators=150,verbose=-1,random_state=42,n_jobs=-1),
            XR(n_estimators=150,verbosity=0,random_state=42,n_jobs=-1)
        ] for _ in range(nh)]
        bc_mh=[[
            CBC(verbose=0,iterations=100,random_seed=42,thread_count=-1),
            LC(n_estimators=100,verbose=-1,random_state=42,n_jobs=-1),
            XC(n_estimators=100,verbosity=0,eval_metric='logloss',random_state=42,n_jobs=-1)
        ] for _ in range(nh)]
        
        S_Z=np.zeros((len(P_target),nh),dtype=np.float64)
        transformers_mh=[self._clone_transformer(transformer) for _ in range(nh)]
        
        for hi_idx in range(nh):
            transformers_mh[hi_idx].fit(self.P[:t,hi_idx])
            S_Z[:t,hi_idx]=transformers_mh[hi_idx].transform(self.P[:t,hi_idx])
        
        t2=int(t*0.7)
        print(f"⏳ Fase 1/3: Meta-features multi-horizon ({t2}→{t-t2} val)...")
        
        [m.fit(Xs[:t2],y_dir[:t2]) for m in bc]
        for hi_idx in range(nh):
            [m.fit(Xs[:t2],S_Z[:t2,hi_idx]) for m in br_all[hi_idx]]
        for hi_idx in range(nh):
            [m.fit(Xs[:t2],self.D[:t2,hi_idx]) for m in bc_mh[hi_idx]]
        
        n_base_cls=3
        n_base_reg=3*nh
        n_mh_cls=3*nh
        n_meta_feat=n_base_cls+n_base_reg+n_mh_cls
        
        Mc0=np.column_stack([m.predict_proba(Xs[t2:t])[:,1] for m in bc])
        Mr0_all=np.column_stack([m.predict(Xs[t2:t]) for hi_idx in range(nh) for m in br_all[hi_idx]])
        Mh0_all=np.column_stack([m.predict_proba(Xs[t2:t])[:,1] for hi_idx in range(nh) for m in bc_mh[hi_idx]])
        
        Meta0_cls=np.column_stack([Mc0,Mr0_all,Mh0_all])
        Meta0_reg=np.column_stack([Mc0,Mr0_all,Mh0_all])
        
        mc=LR(max_iter=2000,n_jobs=-1)
        mc.fit(Meta0_cls,y_dir[t2:t])
        
        # CORREGIDO: seq_len más pequeño y robusto
        seq_len_safe = min(15, max(3, len(Meta0_reg)//10))
        mr=TimesNetMetaWrapper(
            n_features=n_meta_feat,
            seq_len=seq_len_safe,
            lr=1e-3,
            epochs=30,
            d_model=32,
            d_ff=32,
            e_layers=2,
            top_k=2,
            num_kernels=3
        )
        mr.fit(Meta0_reg, S_Z[t2:t, hi])
        
        print(f"⏳ Fase 2/3: Re-entrenamiento base completo ({t} muestras)...")
        [m.fit(Xs[:t],y_dir[:t]) for m in bc]
        for hi_idx in range(nh):
            [m.fit(Xs[:t],S_Z[:t,hi_idx]) for m in br_all[hi_idx]]
        for hi_idx in range(nh):
            [m.fit(Xs[:t],self.D[:t,hi_idx]) for m in bc_mh[hi_idx]]
        
        nt_total=len(y_dir)-t
        test_indices=list(range(0,nt_total,stride))
        nt=len(test_indices)
        
        bs0=t-t2
        bsz=bs0+nt
        Mcb=np.zeros((bsz,n_meta_feat))
        ycb=np.zeros(bsz)
        prb=np.zeros(bsz)
        
        ci0=np.column_stack([m.predict_proba(Xs[t2:t])[:,1] for m in bc])
        ri0=np.column_stack([m.predict(Xs[t2:t]) for hi_idx in range(nh) for m in br_all[hi_idx]])
        mh0=np.column_stack([m.predict_proba(Xs[t2:t])[:,1] for hi_idx in range(nh) for m in bc_mh[hi_idx]])
        
        Mcb[:bs0]=np.column_stack([ci0,ri0,mh0])
        ycb[:bs0]=y_dir[t2:t]
        prb[:bs0]=S_Z[t2:t,hi]
        bs=bs0
        
        pd_a=np.zeros(nt)
        pp_a=np.zeros(nt)
        recent_errors=[]
        
        print(f"⏳ Fase 3/3: Walk-forward adaptivo ({nt} pasos, stride={stride})...")
        last_retrain=0
        
        for step,off in enumerate(test_indices):
            idx=t+off
            if idx>=len(Xs):
                break
            
            xi=Xs[idx:idx+1]
            ci=np.array([m.predict_proba(xi)[:,1][0] for m in bc])
            ri=np.concatenate([np.array([m.predict(xi)[0] for m in br_all[hi_idx]]) for hi_idx in range(nh)])
            mhi=np.concatenate([np.array([m.predict_proba(xi)[:,1][0] for m in bc_mh[hi_idx]]) for hi_idx in range(nh)])
            meta_feat=np.concatenate([ci,ri,mhi]).reshape(1,-1)
            
            pd_a[step]=mc.predict(meta_feat)[0]
            z_pred=mr.predict(meta_feat)[0]
            pp_a[step]=transformers_mh[hi].inverse_transform(np.array([z_pred]))[0]
            
            actual_Z=transformers_mh[hi].transform(np.array([self.P[idx,hi]]))[0]
            S_Z[idx,hi]=actual_Z
            
            for hi_idx in range(nh):
                if hi_idx!=hi:
                    S_Z[idx,hi_idx]=transformers_mh[hi_idx].transform(np.array([self.P[idx,hi_idx]]))[0]
            
            Mcb[bs]=meta_feat.ravel()
            ycb[bs]=y_dir[idx]
            prb[bs]=actual_Z
            bs+=1
            
            err=abs(pp_a[step]-self.P[idx,hi])
            recent_errors.append(err)
            if len(recent_errors)>self.err_window:
                recent_errors.pop(0)
            
            sw=np.exp(np.linspace(-3,0,bs))
            mc.fit(Mcb[:bs],ycb[:bs],sample_weight=sw)
            mr.fit(Mcb[:bs],prb[:bs],sample_weight=sw)
            
            need_retrain=False
            if len(recent_errors)>=self.err_window:
                med_err=np.median(recent_errors)
                baseline=np.median(np.abs(np.diff(self.P[max(0,idx-50):idx,hi]))) if idx>50 else med_err
                if baseline>0 and med_err/baseline>self.err_thresh:
                    need_retrain=True
                    print(f"  ⚡ Régimen detectado step={step} err_ratio={med_err/baseline:.2f}")
            
            if need_retrain or (step-last_retrain>=50 and step>0):
                e=idx+1
                last_retrain=step
                for hi_idx in range(nh):
                    transformers_mh[hi_idx].fit(self.P[:e,hi_idx])
                    S_Z[:e,hi_idx]=transformers_mh[hi_idx].transform(self.P[:e,hi_idx])
                [m.fit(Xs[:e],y_dir[:e]) for m in bc]
                for hi_idx in range(nh):
                    [m.fit(Xs[:e],S_Z[:e,hi_idx]) for m in br_all[hi_idx]]
                for hi_idx in range(nh):
                    [m.fit(Xs[:e],self.D[:e,hi_idx]) for m in bc_mh[hi_idx]]
                recent_errors.clear()
                print(f"  🔄 Reentrenamiento en step={step}")
        
        ye=y_dir[t:t+nt_total:stride][:nt]
        pe=self.P[t:t+nt_total:stride,hi][:nt]
        a=AC(ye,pd_a[:len(ye)])
        r2=RS(pe,pp_a[:len(pe)])
        
        print(f"✅ Dirección Acc: {a:.4f} │ Precio R²: {r2:.4f}")
        self.graficar(pe,pp_a[:len(pe)],ye,pd_a[:len(ye)],r2,a,t_name)
        return a,r2

    def _clone_transformer(self,t):
        import copy
        try:
            return copy.deepcopy(t)
        except:
            return type(t)() if not isinstance(t,Transf_PyTorch_Flow) else Transf_PyTorch_Flow()

    def graficar(self,pe,pp_a,ye,pd_a,r2,a,t_name):
        plt.style.use('default')
        plt.rcParams.update({'figure.facecolor':'w','axes.facecolor':'w','axes.grid':True,'grid.alpha':.25})
        fig,(a1,a2)=plt.subplots(2,1,figsize=(16,10),gridspec_kw={'hspace':.4})
        fig.patch.set_facecolor('w')
        n=len(ye)
        ok=(pd_a==ye)
        
        a1.plot(pe,c='#1565c0',lw=1.3,label='Real')
        a1.plot(pp_a,c='#c62828',lw=1.1,label='Predicho',alpha=.8)
        a1.fill_between(range(n),pe,pp_a,alpha=.08,color='#e65100')
        a1.set_title(f'[{t_name}] {self.tk} Precio │ TimesNet Meta │ R²={r2:.4f} │ stride={self.stride}',fontsize=13,fontweight='bold')
        a1.legend(fontsize=11)
        a1.set_ylabel('Precio ($)',fontsize=11)
        
        a2.fill_between(range(n),-.15,1.15,where=ok,color='#2e7d32',alpha=.1,label='Acierto')
        a2.fill_between(range(n),-.15,1.15,where=~ok,color='#c62828',alpha=.1,label='Error')
        a2.step(range(n),ye,c='#1565c0',lw=1.5,where='mid',label='Real')
        a2.step(range(n),pd_a,c='#c62828',lw=1.1,where='mid',label='Predicho',alpha=.7)
        a2.set_title(f'[{t_name}] {self.tk} Dirección │ Acc={a:.4f}',fontsize=13,fontweight='bold')
        a2.legend(fontsize=10,ncol=4,loc='upper center')
        a2.set_ylabel('0=Baja 1=Sube',fontsize=11)
        a2.set_ylim(-.2,1.2)
        a2.set_yticks([0,1])
        
        plt.xlabel('Muestras Test',fontsize=11)
        plt.savefig(f'{self.tk}_{t_name}_adaptive.png',dpi=150,bbox_inches='tight',facecolor='w')
        plt.close()

    def generar_html(self,resultados):
        html="""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Reporte Stacking Multi-Horizon TimesNet</title>
<style>body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background-color:#f0f2f5;padding:40px}
.container{max-width:1200px;margin:auto;background:white;padding:30px;border-radius:8px;box-shadow:0 4px 8px rgba(0,0,0,0.1)}
h1{color:#333;text-align:center}table{width:100%;border-collapse:collapse;margin-top:20px}
th,td{padding:12px;text-align:center;border-bottom:1px solid #ddd}th{background-color:#1565c0;color:white}
tr:hover{background-color:#f5f5f5}.img-section{margin-top:40px}
.img-card{margin-bottom:40px;padding:20px;background:#fafafa;border-radius:8px}
.img-card h2{color:#1565c0;margin-top:0}.img-card img{max-width:100%;height:auto;display:block;margin:20px auto}
.metrics{display:flex;justify-content:space-around;margin:15px 0}
.metric-box{background:#1565c0;color:white;padding:15px 25px;border-radius:6px;text-align:center}
.metric-box .value{font-size:24px;font-weight:bold}.metric-box .label{font-size:12px;opacity:0.9}</style></head>
<body><div class="container"><h1>Reporte Multi-Horizon + TimesNet Meta-Learner</h1>
<p style="text-align:center;color:#666;">stride=h (sin solapamiento) | multi-horizon [1,3,5,10] | reentrenamiento adaptivo por error</p>
<table><tr><th>Transformación</th><th>Accuracy (Dir)</th><th>R² (Precio)</th></tr>"""
        for r in resultados:
            html+=f"<tr><td>{r[0]}</td><td>{r[1]:.4f}</td><td>{r[2]:.4f}</td></tr>"
        html+="</table><div class='img-section'><h2 style='text-align:center;color:#333;'>Visualizaciones</h2>"
        for name,acc,r2 in resultados:
            html+=f"<div class='img-card'><h2>{name}</h2><div class='metrics'><div class='metric-box'><div class='value'>{acc:.4f}</div><div class='label'>Accuracy</div></div><div class='metric-box'><div class='value'>{r2:.4f}</div><div class='label'>R²</div></div></div><img src='{self.tk}_{name}_adaptive.png' alt='{name}'></div>"
        html+="</div></div></body></html>"
        with open("metricas_transformaciones.html","w",encoding='utf-8') as f:
            f.write(html)
        print("\n✅ Reporte HTML generado: 'metricas_transformaciones.html'")

if __name__=="__main__":
    modelo=VGSS(tk="AAPL",horizons=[1,3,5,10])
    modelo.run_all()



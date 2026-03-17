import numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt
import torch, torch.nn as nn
from numba import njit
from scipy.special import logit, expit
from scipy.interpolate import PchipInterpolator
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler as SS, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LogisticRegression as LR, Ridge as RD
from sklearn.metrics import accuracy_score as AC, r2_score as RS
from catboost import CatBoostClassifier as CBC, CatBoostRegressor as CBR
from lightgbm import LGBMClassifier as LC, LGBMRegressor as LGR
from xgboost import XGBClassifier as XC, XGBRegressor as XR
import warnings; warnings.filterwarnings('ignore')

yf.enable_debug_mode()

# =========================================================================
# 1. FUNCIONES NUMBA (Base features - Optimizadas en paralelo)
# =========================================================================
@njit(fastmath=False, cache=True)
def vg(s):
    N = len(s)
    ki = np.zeros(N, dtype=np.float64)
    ko = np.zeros(N, dtype=np.float64)
    for i in range(N-1):
        m = -1e15
        for j in range(i+1, N):
            denom = float(j - i)
            if denom > 0:
                p = (s[j] - s[i]) / denom
                if p > m:
                    ko[i] += 1.0
                    ki[j] += 1.0
                    m = p
    return ki, ko, ki + ko

@njit(fastmath=False, cache=True)
def kld(ki, ko):
    N = len(ki)
    mx = int(max(np.max(ki), np.max(ko))) + 1
    ci = np.zeros(mx, dtype=np.float64)
    co = np.zeros(mx, dtype=np.float64)
    for i in range(N):
        ki_idx = int(ki[i])
        ko_idx = int(ko[i])
        if 0 <= ki_idx < mx:
            ci[ki_idx] += 1.0
        if 0 <= ko_idx < mx:
            co[ko_idx] += 1.0
    pi = (ci / N) + (1.0 / N)
    po = (co / N) + (1.0 / N)
    pi = pi / np.sum(pi)
    po = po / np.sum(po)
    result = 0.0
    for i in range(len(pi)):
        if pi[i] > 0 and po[i] > 0:
            result += pi[i] * np.log(pi[i] / po[i])
    return result

@njit(fastmath=False, cache=True)
def hs(kt):
    c = np.zeros(int(np.max(kt)) + 1, dtype=np.float64)
    for v in kt:
        v_idx = int(v)
        if 0 <= v_idx < len(c):
            c[v_idx] += 1.0
    k = np.nonzero(c)[0]
    p = c[k] / len(kt)
    m = (k >= 2) & (p > 0)
    if np.sum(m) < 3:
        return 0.5
    x = np.log(k[m].astype(np.float64))
    y = np.log(p[m])
    n = len(x)
    sx = np.sum(x)
    h = (3 + (n * np.sum(x * y) - sx * np.sum(y)) / (n * np.sum(x * x) - sx ** 2)) / 2
    return max(0.01, min(0.99, h))

@njit(fastmath=False, cache=True)
def fvg(s):
    ki, ko, kt = vg(s)
    W = len(s)
    nb = max(3, W // 10)
    nt = min(10, W // 3)
    gr = np.mean(kt[-nb:])
    ga = max(1.0, np.mean(kt[:nb]))
    sl = 0.0
    if nt >= 3:
        i = np.arange(nt, dtype=np.float64)
        denom = nt * np.sum(i ** 2) - np.sum(i) ** 2
        if denom != 0:
            sl = (nt * np.sum(i * ki[-nt:]) - np.sum(i) * np.sum(ki[-nt:])) / denom
    m1 = np.mean(kt)
    m2 = np.mean((kt - m1) ** 2)
    unique_vals = np.unique(kt)
    pk = np.array([np.sum(kt == v) for v in unique_vals]) / W
    skew = 0.0
    if m2 > 0:
        skew = np.mean((kt - m1) ** 3) / (m2 ** 1.5)
    entropy = 0.0
    for p in pk:
        if p > 0:
            entropy -= p * np.log(p + 1e-12)
    return np.array([
        ki[-1], ki[-1] / max(1.0, np.max(kt)),
        kld(ki, ko), hs(kt), m1, skew,
        gr / ga, sl, np.max(kt), np.std(kt) / max(m1, 1.0),
        np.mean(ko[-nb:] - ki[-nb:]), entropy
    ])

@njit(fastmath=False, cache=True)
def ftr(s):
    r = np.diff(s) / s[:-1]
    W = len(s)
    mr = np.mean(r)
    v2 = np.mean((r - mr) ** 2)
    d = np.diff(s[-15:]) if W >= 15 else np.array([0.0])
    md = np.mean(np.maximum(d, 0.0))
    mu = np.mean(np.maximum(-d, 0.0))
    dd_ratio = 100.0 if W < 15 or mu == 0 else 100.0 - 100.0 / (1.0 + md / mu)
    price_range = max(np.max(s) - np.min(s), 1e-8)
    ma_ref = np.mean(s[-20:]) if W >= 20 else s[-1]
    return np.array([
        s[-1] / s[0] - 1,
        np.std(r),
        np.mean((r - mr) ** 3) / (v2 ** 1.5) if v2 > 0 else 0.0,
        np.mean((r - mr) ** 4) / (v2 ** 2) if v2 > 0 else 0.0,
        dd_ratio,
        (s[-1] - np.min(s)) / price_range,
        s[-1] / ma_ref - 1
    ])

@njit(parallel=False, fastmath=False, cache=True)
def bds(p, v=40, h=5):
    M = len(p) - v - h
    X = np.zeros((M, 19), dtype=np.float64)
    D = np.zeros(M, dtype=np.float64)
    R = np.zeros(M, dtype=np.float64)
    P = np.zeros(M, dtype=np.float64)
    for i in range(M):
        fv = fvg(p[i:i + v + 1])
        tr = ftr(p[i:i + v + 1])
        for j in range(12):
            X[i, j] = fv[j]
        for j in range(7):
            X[i, 12 + j] = tr[j]
        R[i] = (p[i + v + h] - p[i + v]) / p[i + v]
        D[i] = 1.0 if R[i] > 0 else 0.0
        P[i] = p[i + v + h]
    return X, D, R, P


# =========================================================================
# 2. TRANSFORMACIONES MATEMÁTICAS (Aisladas y estables para CPU)
# =========================================================================

class Transf_PIT_Logit:
    """ 1. Transformación PIT Empírica + Logit """
    def fit(self, y): self.qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100).fit(y.reshape(-1,1))
    def transform(self, y): return logit(np.clip(self.qt.transform(y.reshape(-1,1)).ravel(), 1e-6, 1-1e-6))
    def inverse_transform(self, z): return self.qt.inverse_transform(expit(z).reshape(-1,1)).ravel()

class Transf_PyTorch_Flow:
    """ 2. Normalizing Flow proxy (Affine Coupling Neural Flow con PyTorch) """
    def __init__(self):
        self.w = nn.Parameter(torch.tensor(1.0)); self.b = nn.Parameter(torch.tensor(0.0))
        self.opt = torch.optim.Adam([self.w, self.b], lr=0.1)
    def fit(self, y):
        yt = torch.tensor(y, dtype=torch.float32)
        for _ in range(100):
            self.opt.zero_grad()
            z = yt * self.w + self.b
            loss = torch.mean(0.5 * z**2) - torch.log(torch.abs(self.w) + 1e-6) # NLL
            loss.backward(); self.opt.step()
    def transform(self, y): return (torch.tensor(y, dtype=torch.float32) * self.w + self.b).detach().numpy()
    def inverse_transform(self, z): return ((torch.tensor(z, dtype=torch.float32) - self.b) / self.w).detach().numpy()

class Transf_Spline:
    """ 3. Spline Racional Proxy (PCHIP Interpolation) """
    def fit(self, y):
        self.q = np.linspace(0, 1, 15)
        self.y_q = np.quantile(y, self.q)
        self.z_q = QuantileTransformer(output_distribution='normal').fit_transform(self.y_q.reshape(-1,1)).ravel()
        self.z_q[0], self.z_q[-1] = -3.5, 3.5 
        self.fwd_spline = PchipInterpolator(self.y_q, self.z_q)
        self.inv_spline = PchipInterpolator(self.z_q, self.y_q)
    def transform(self, y): return self.fwd_spline(np.clip(y, self.y_q[0], self.y_q[-1]))
    def inverse_transform(self, z): return self.inv_spline(np.clip(z, self.z_q[0], self.z_q[-1]))

class Transf_INN_Power:
    """ 4. INN BoxCox/Yeo-Johnson (Biyectores de potencia) """
    def fit(self, y): self.pt = PowerTransformer(method='yeo-johnson', standardize=True).fit(y.reshape(-1,1))
    def transform(self, y): return self.pt.transform(y.reshape(-1,1)).ravel()
    def inverse_transform(self, z): return self.pt.inverse_transform(z.reshape(-1,1)).ravel()

class Transf_Sktime_Pipeline:
    """ 5. Pipeline proxy sktime (Log -> StandardScaler) """
    def fit(self, y): self.ss = SS().fit(np.log(y+1e-8).reshape(-1,1))
    def transform(self, y): return self.ss.transform(np.log(y+1e-8).reshape(-1,1)).ravel()
    def inverse_transform(self, z): return np.exp(self.ss.inverse_transform(z.reshape(-1,1)).ravel())

class Transf_Omni_Custom:
    """ 6. Custom Omni-Transform: Quantile -> SinhArcsinh (Modifica curtosis suavemente) """
    def fit(self, y): self.qt = QuantileTransformer(output_distribution='normal', n_quantiles=150).fit(y.reshape(-1,1))
    def transform(self, y): return np.sinh(self.qt.transform(y.reshape(-1,1)).ravel())
    def inverse_transform(self, z): return self.qt.inverse_transform(np.arcsinh(z).reshape(-1,1)).ravel()


# =========================================================================
# 3. PIPELINE DE STACKING EVALUANDO MULTIPLES TRANSFORMACIONES
# =========================================================================
class VGSS:
    def __init__(self, tk="AAPL", v=40, h=5, rt=0.7):
        print(f"📥 Descargando datos para {tk}...")
        d_train = yf.download(tk, start="2015-01-01", end="2025-12-31", progress=False)
        d_test = yf.download(tk, start="2026-01-01", progress=False)
        if isinstance(d_train.columns, pd.MultiIndex):
            p1 = np.ascontiguousarray(d_train[('Close',tk)].values.ravel().astype(np.float64))
            p2 = np.ascontiguousarray(d_test[('Close',tk)].values.ravel().astype(np.float64))
        else:
            p1 = np.ascontiguousarray(d_train['Close'].values.ravel().astype(np.float64))
            p2 = np.ascontiguousarray(d_test['Close'].values.ravel().astype(np.float64))
        
        self.p = np.concatenate([p1, p2]); self.tk = tk
        print("⚙️ Generando features matemáticos...")
        self.X, self.y, self.R, self.P = bds(self.p, v, h)
        self.tr = int(len(self.y) * rt)

    def run_all(self):
        transformaciones = {
            "PIT_Aprendida_Logit": Transf_PIT_Logit(),
            "Normalizing_Flow_PyTorch": Transf_PyTorch_Flow(),
            "Rational_Spline_PCHIP": Transf_Spline(),
            "INN_Yeo_Johnson": Transf_INN_Power(),
            "Sktime_Pipeline_LogSS": Transf_Sktime_Pipeline(),
            "Omni_Custom_SinhArcsinh": Transf_Omni_Custom()
        }

        resultados = []
        for name, transformer in transformaciones.items():
            print(f"\n{'='*70}\n🚀 Evaluando Transformación: {name}\n{'='*70}")
            a, r2 = self.run_single_transform(name, transformer)
            resultados.append((name, a, r2))
        
        self.generar_html(resultados)

    def run_single_transform(self, t_name, transformer):
        Xs = SS().fit_transform(self.X); t, nt, RE = self.tr, len(self.y)-self.tr, 50
        
        # Parámetros pesados de CPU activados (n_jobs=-1, thread_count=-1)
        bc = [CBC(verbose=0, iterations=150, random_seed=42, thread_count=-1),
              LC(n_estimators=150, verbose=-1, random_state=42, n_jobs=-1),
              XC(n_estimators=150, verbosity=0, eval_metric='logloss', random_state=42, n_jobs=-1)]
        
        br = [CBR(verbose=0, iterations=150, random_seed=42, thread_count=-1),
              LGR(n_estimators=150, verbose=-1, random_state=42, n_jobs=-1),
              XR(n_estimators=150, verbosity=0, random_state=42, n_jobs=-1)]
        
        # Preparar Target Transformado (solo con datos de train)
        S_Z = np.zeros_like(self.P)
        transformer.fit(self.P[:t])
        S_Z[:t] = transformer.transform(self.P[:t])

        t2 = int(t * 0.7)
        print(f"⏳ Fase 1/3: Meta-features iniciales ({t2}→{t-t2} val)...")
        [m.fit(Xs[:t2], self.y[:t2]) for m in bc]
        [m.fit(Xs[:t2], S_Z[:t2]) for m in br] # Entrenar con Z

        Mc0 = np.column_stack([m.predict_proba(Xs[t2:t])[:,1] for m in bc])
        Mr0 = np.column_stack([m.predict(Xs[t2:t]) for m in br])
        
        mc = LR(max_iter=2000, n_jobs=-1); mr = RD()
        mc.fit(Mc0, self.y[t2:t]); mr.fit(Mr0, S_Z[t2:t])
        
        print(f"⏳ Fase 2/3: Base completo ({t} muestras)...")
        [m.fit(Xs[:t], self.y[:t]) for m in bc]
        [m.fit(Xs[:t], S_Z[:t]) for m in br]
        
        bs0 = t - t2; bsz = bs0 + nt
        Mcb, Mrb, ycb, prb = np.zeros((bsz,3)), np.zeros((bsz,3)), np.zeros(bsz), np.zeros(bsz)
        Mcb[:bs0], Mrb[:bs0], ycb[:bs0], prb[:bs0] = Mc0, Mr0, self.y[t2:t], S_Z[t2:t]; bs = bs0
        
        pd_a, pp_a = np.zeros(nt), np.zeros(nt)
        print(f"⏳ Fase 3/3: Walk-forward adaptivo ({nt} pasos)...")
        
        for i in range(nt):
            xi = Xs[t+i:t+i+1]
            ci = np.array([m.predict_proba(xi)[:,1][0] for m in bc])
            ri = np.array([m.predict(xi)[0] for m in br])
            
            pd_a[i] = mc.predict(ci.reshape(1,-1))[0]
            z_pred  = mr.predict(ri.reshape(1,-1))[0]
            
            # Inversión de la predicción en el espacio real P
            pp_a[i] = transformer.inverse_transform(np.array([z_pred]))[0]
            
            # Guardar el real en el espacio latente para el Ridge
            actual_Z = transformer.transform(np.array([self.P[t+i]]))[0]
            S_Z[t+i] = actual_Z

            Mcb[bs], Mrb[bs], ycb[bs], prb[bs] = ci, ri, self.y[t+i], actual_Z
            bs += 1
            
            sw = np.exp(np.linspace(-3, 0, bs))
            mc.fit(Mcb[:bs], ycb[:bs], sample_weight=sw)
            mr.fit(Mrb[:bs], prb[:bs], sample_weight=sw)
            
            if (i+1) % RE == 0:
                e = t + i + 1
                transformer.fit(self.P[:e]) # Actualizar conocimientos del transformador
                S_Z[:e] = transformer.transform(self.P[:e]) # Re-transformar historia profunda
                [m.fit(Xs[:e], self.y[:e]) for m in bc]
                [m.fit(Xs[:e], S_Z[:e]) for m in br]

        ye, pe = self.y[t:], self.P[t:]
        a, r2 = AC(ye, pd_a), RS(pe, pp_a)
        
        print(f"✅ Dirección Acc: {a:.4f} │ Precio R²: {r2:.4f}")
        self.graficar(pe, pp_a, ye, pd_a, r2, a, t_name)
        return a, r2

    def graficar(self, pe, pp_a, ye, pd_a, r2, a, t_name):
        plt.style.use('default'); plt.rcParams.update({'figure.facecolor':'w','axes.facecolor':'w','axes.grid':True,'grid.alpha':.25})
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace':.4}); fig.patch.set_facecolor('w')
        n = len(ye); ok = (pd_a == ye)
        
        a1.plot(pe, c='#1565c0', lw=1.3, label='Real')
        a1.plot(pp_a, c='#c62828', lw=1.1, label='Predicho', alpha=.8)
        a1.fill_between(range(n), pe, pp_a, alpha=.08, color='#e65100')
        a1.set_title(f'[{t_name}] {self.tk} Precio Cierre │ Stacking Adaptivo │ R²={r2:.4f}', fontsize=13, fontweight='bold')
        a1.legend(fontsize=11); a1.set_ylabel('Precio ($)', fontsize=11)
        
        a2.fill_between(range(n), -.15, 1.15, where=ok, color='#2e7d32', alpha=.1, label='Acierto')
        a2.fill_between(range(n), -.15, 1.15, where=~ok, color='#c62828', alpha=.1, label='Error')
        a2.step(range(n), ye, c='#1565c0', lw=1.5, where='mid', label='Real')
        a2.step(range(n), pd_a, c='#c62828', lw=1.1, where='mid', label='Predicho', alpha=.7)
        a2.set_title(f'[{t_name}] {self.tk} Dirección │ LogReg │ Acc={a:.4f}', fontsize=13, fontweight='bold')
        a2.legend(fontsize=10, ncol=4, loc='upper center'); a2.set_ylabel('0=Baja 1=Sube', fontsize=11); a2.set_ylim(-.2, 1.2); a2.set_yticks([0,1])
        
        plt.xlabel('Muestras Test', fontsize=11)
        plt.savefig(f'{self.tk}_{t_name}_adaptive.png', dpi=150, bbox_inches='tight', facecolor='w')
        plt.close()

    def generar_html(self, resultados):
        html = """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Reporte Stacking Transformations</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; padding: 40px; }
                .container { max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }
                th { background-color: #1565c0; color: white; }
                tr:hover { background-color: #f5f5f5; }
                .best { font-weight: bold; color: #2e7d32; }
                .img-section { margin-top: 40px; }
                .img-card { margin-bottom: 40px; padding: 20px; background: #fafafa; border-radius: 8px; }
                .img-card h2 { color: #1565c0; margin-top: 0; }
                .img-card img { max-width: 100%; height: auto; display: block; margin: 20px auto; }
                .metrics { display: flex; justify-content: space-around; margin: 15px 0; }
                .metric-box { background: #1565c0; color: white; padding: 15px 25px; border-radius: 6px; text-align: center; }
                .metric-box .value { font-size: 24px; font-weight: bold; }
                .metric-box .label { font-size: 12px; opacity: 0.9; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Reporte de Métricas: Transformaciones Latentes</h1>
                <table>
                    <tr><th>Transformación Utilizada</th><th>Accuracy (Dirección)</th><th>R² (Precio Forecasting)</th></tr>
        """
        for r in resultados:
            html += f"<tr><td>{r[0]}</td><td>{r[1]:.4f}</td><td>{r[2]:.4f}</td></tr>"

        html += """
                </table>
                <p style="text-align:center; margin-top:30px; color:#666;">Modelos base: LightGBM, XGBoost, CatBoost | CPU Optimized</p>
                
                <div class="img-section">
                    <h2 style="text-align:center; color:#333;">Visualizaciones por Transformación</h2>
        """
        
        for name, acc, r2 in resultados:
            img_file = f"{self.tk}_{name}_adaptive.png"
            html += f"""
                    <div class="img-card">
                        <h2>{name}</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <div class="value">{acc:.4f}</div>
                                <div class="label">Accuracy (Dirección)</div>
                            </div>
                            <div class="metric-box">
                                <div class="value">{r2:.4f}</div>
                                <div class="label">R² (Precio Forecasting)</div>
                            </div>
                        </div>
                        <img src="{img_file}" alt="{name} - Gráfico Adaptivo">
                    </div>
            """

        html += """
                </div>
            </div>
        </body>
        </html>
        """
        with open("metricas_transformaciones.html", "w", encoding='utf-8') as f:
            f.write(html)
        print("\n✅ Reporte HTML generado exitosamente: 'metricas_transformaciones.html'")

if __name__ == "__main__":
    modelo = VGSS(tk="AAPL")
    modelo.run_all()


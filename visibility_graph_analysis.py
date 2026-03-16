"""
=============================================================================
ANÁLISIS DE VISIBILITY GRAPH - IRREVERSIBILIDAD TEMPORAL Y FRACTALES
Aplicado al precio de cierre de Apple (AAPL) 2020-2025

Basado en: Xiong, H., Shang, P., Xia, J., Wang, J. (2018).
"Time irreversibility and intrinsics revealing of series with complex
network approach", Physica A.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

# === Instalar si es necesario: pip install yfinance ===
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("⚠ yfinance no instalado. Ejecuta: pip install yfinance")


# ====================================================================
# PASO 1: DESCARGAR DATOS
# ====================================================================
def descargar_datos(ticker="AAPL", inicio="2020-01-01", fin="2025-01-01"):
    """Descarga precios de cierre desde Yahoo Finance."""
    if HAS_YF:
        df = yf.download(ticker, start=inicio, end=fin, progress=False)
        precios = df['Close'].values.flatten()
        fechas = df.index
        return precios, fechas
    else:
        # Simulación tipo random walk si no hay yfinance
        np.random.seed(42)
        N = 1260
        retornos = np.random.normal(0.0005, 0.018, N)
        precios = 75 * np.exp(np.cumsum(retornos))
        fechas = pd.date_range(inicio, periods=N, freq='B')
        return precios, fechas


# ====================================================================
# PASO 2: CONSTRUIR EL VISIBILITY GRAPH
# ====================================================================
def construir_visibility_graph(serie, guardar_aristas=True):
    """
    Construye el Visibility Graph dirigido de una serie temporal.

    Algoritmo O(N²) optimizado:
    Desde cada nodo i, escaneamos hacia la derecha manteniendo la
    pendiente máxima vista. Un nodo j es visible desde i si y solo si
    la pendiente de i a j supera esa pendiente máxima.

    Parámetros
    ----------
    serie : array de N valores reales
    guardar_aristas : bool, si True guarda la lista de aristas

    Retorna
    -------
    aristas : lista de tuplas (i, j) con i < j  [si guardar_aristas=True]
    k_in    : array[N] - grado de entrada de cada nodo
    k_out   : array[N] - grado de salida de cada nodo
    k_total : array[N] - grado total
    """
    N = len(serie)
    aristas = [] if guardar_aristas else None
    k_in = np.zeros(N, dtype=int)
    k_out = np.zeros(N, dtype=int)

    for i in range(N - 1):
        pendiente_max = -np.inf
        for j in range(i + 1, N):
            # Pendiente de la línea de visión desde i hasta j
            pendiente = (serie[j] - serie[i]) / (j - i)

            if pendiente > pendiente_max:
                # ¡j es visible desde i!
                if guardar_aristas:
                    aristas.append((i, j))
                k_out[i] += 1   # Arista SALIENTE para i (mira al futuro)
                k_in[j] += 1    # Arista ENTRANTE para j (vista desde el pasado)
                pendiente_max = pendiente

    k_total = k_in + k_out

    if guardar_aristas:
        return aristas, k_in, k_out, k_total
    else:
        return k_in, k_out, k_total


# ====================================================================
# PASO 3: DISTRIBUCIONES DE GRADO
# ====================================================================
def distribucion_grado(grados):
    """
    Calcula P(k) = fracción de nodos con grado = k.
    Retorna los valores de k y sus probabilidades.
    """
    conteo = Counter(grados)
    N = len(grados)
    k_vals = np.array(sorted(conteo.keys()))
    p_vals = np.array([conteo[k] / N for k in k_vals])
    return k_vals, p_vals


# ====================================================================
# PASO 4: DIVERGENCIA DE KULLBACK-LEIBLER (KLD)
# ====================================================================
def calcular_kld(k_in, k_out, gamma=1):
    """
    Calcula la KLD entre distribuciones de grado entrante y saliente.

    D_kld(P_in || P_out) = Σ_k P_in(k) · log(P_in(k) / P_out(k))

    Se aplica suavizado (sesgo γ/N) para evitar divisiones por cero.

    KLD ≈ 0: serie reversible en el tiempo
    KLD >> 0: serie irreversible en el tiempo
    """
    N = len(k_in)
    conteo_in = Counter(k_in)
    conteo_out = Counter(k_out)

    # Todos los valores posibles de k
    todos_k = sorted(set(list(conteo_in.keys()) + list(conteo_out.keys())))

    epsilon = gamma / N  # Sesgo pequeño

    # Construir distribuciones con suavizado
    p_in = np.array([conteo_in.get(k, 0) / N + epsilon for k in todos_k])
    p_out = np.array([conteo_out.get(k, 0) / N + epsilon for k in todos_k])

    # Renormalizar para que sumen 1
    p_in = p_in / p_in.sum()
    p_out = p_out / p_out.sum()

    # KLD
    kld = np.sum(p_in * np.log(p_in / p_out))
    return kld


# ====================================================================
# PASO 5: AJUSTE DE LEY DE POTENCIA Y HURST
# ====================================================================
def ajustar_ley_potencia(k_vals, p_vals, k_min=2):
    """
    Ajusta P(k) ~ k^{-r} por regresión log-log.
    Estima H = (3 - r) / 2 (relación válida para procesos tipo fBm).
    """
    mascara = (k_vals >= k_min) & (p_vals > 0)
    if mascara.sum() < 3:
        return np.nan, np.nan

    log_k = np.log(k_vals[mascara].astype(float))
    log_p = np.log(p_vals[mascara])

    pendiente, intercepto, r_val, p_val, std_err = stats.linregress(log_k, log_p)

    r = -pendiente              # Exponente de ley de potencia
    H = (3 - r) / 2             # Exponente de Hurst estimado
    r_squared = r_val ** 2      # Bondad del ajuste

    return r, H, r_squared


# ====================================================================
# VERIFICACIÓN: VG por fuerza bruta (para validar el algoritmo)
# ====================================================================
def vg_fuerza_bruta(serie):
    """
    Implementación O(N³) para verificar la corrección del algoritmo rápido.
    Solo usar con series cortas (N < 100).
    """
    N = len(serie)
    aristas = []
    for i in range(N):
        for j in range(i + 1, N):
            visible = True
            for k in range(i + 1, j):
                umbral = serie[i] + (serie[j] - serie[i]) * (k - i) / (j - i)
                if serie[k] >= umbral:
                    visible = False
                    break
            if visible:
                aristas.append((i, j))
    return aristas


# ====================================================================
# PROGRAMA PRINCIPAL
# ====================================================================
if __name__ == "__main__":

    # ---------------------------------------------------------------
    # 1. DESCARGAR DATOS DE APPLE
    # ---------------------------------------------------------------
    print("=" * 65)
    print("  VISIBILITY GRAPH ANALYSIS — APPLE (AAPL) 2020-2025")
    print("=" * 65)

    precios, fechas = descargar_datos("AAPL", "2020-01-01", "2025-01-01")
    N = len(precios)

    print(f"\n📊 Datos cargados: {N} días de trading")
    print(f"   Período:    {fechas[0].strftime('%Y-%m-%d')} → "
          f"{fechas[-1].strftime('%Y-%m-%d')}")
    print(f"   Precio mín: ${precios.min():.2f}")
    print(f"   Precio máx: ${precios.max():.2f}")
    print(f"   Precio final: ${precios[-1]:.2f}")

    # ---------------------------------------------------------------
    # 2. VERIFICAR ALGORITMO (con los primeros 20 datos)
    # ---------------------------------------------------------------
    print("\n🔍 Verificando algoritmo con primeros 20 datos...")
    sub = precios[:20]
    aristas_bruta = vg_fuerza_bruta(sub)
    aristas_rapida, _, _, _ = construir_visibility_graph(sub)

    if set(aristas_bruta) == set(aristas_rapida):
        print("   ✅ Algoritmo O(N²) verificado: coincide con fuerza bruta")
    else:
        print("   ❌ DISCREPANCIA detectada")
        print(f"      Fuerza bruta: {len(aristas_bruta)} aristas")
        print(f"      Optimizado:   {len(aristas_rapida)} aristas")

    # ---------------------------------------------------------------
    # 3. CONSTRUIR VG COMPLETO
    # ---------------------------------------------------------------
    print(f"\n🔨 Construyendo Visibility Graph ({N} nodos)...")
    t0 = time.time()
    aristas, k_in, k_out, k_total = construir_visibility_graph(precios)
    t_vg = time.time() - t0
    print(f"   ⏱ Tiempo: {t_vg:.1f} segundos")
    print(f"   Aristas totales: {len(aristas)}")
    print(f"   Grado medio:     {k_total.mean():.2f}")
    print(f"   Grado máximo:    {k_total.max()}")
    print(f"   Grado mínimo:    {k_total.min()}")

    # ---------------------------------------------------------------
    # 4. DISTRIBUCIONES DE GRADO
    # ---------------------------------------------------------------
    k_in_vals, P_in = distribucion_grado(k_in)
    k_out_vals, P_out = distribucion_grado(k_out)
    k_total_vals, P_total = distribucion_grado(k_total)

    print(f"\n   Distribución P_in:  {len(k_in_vals)} valores distintos "
          f"(k = {k_in_vals.min()} ... {k_in_vals.max()})")
    print(f"   Distribución P_out: {len(k_out_vals)} valores distintos "
          f"(k = {k_out_vals.min()} ... {k_out_vals.max()})")
    print(f"   Distribución P(k):  {len(k_total_vals)} valores distintos "
          f"(k = {k_total_vals.min()} ... {k_total_vals.max()})")

    # ---------------------------------------------------------------
    # 5. IRREVERSIBILIDAD TEMPORAL (KLD)
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ANÁLISIS DE IRREVERSIBILIDAD TEMPORAL")
    print("=" * 65)

    kld_original = calcular_kld(k_in, k_out)
    print(f"\n   KLD(P_in || P_out) = {kld_original:.6f}")

    # Comparar con datos reshuffled
    print("\n   Calculando KLD para datos aleatorios (control)...")
    kld_aleatorios = []
    for semilla in range(3):
        np.random.seed(semilla + 100)
        aleatorio = precios.copy()
        np.random.shuffle(aleatorio)
        kin_a, kout_a, _ = construir_visibility_graph(aleatorio, guardar_aristas=False)
        kld_a = calcular_kld(kin_a, kout_a)
        kld_aleatorios.append(kld_a)
        print(f"      Shuffle #{semilla+1}: KLD = {kld_a:.6f}")

    kld_ref = np.mean(kld_aleatorios)
    print(f"\n   KLD original:           {kld_original:.6f}")
    print(f"   KLD reshuffled (media): {kld_ref:.6f}")
    print(f"   Ratio:                  {kld_original/kld_ref:.1f}x")

    if kld_original > 3 * kld_ref:
        print("\n   ✅ La serie de Apple es TEMPORALMENTE IRREVERSIBLE")
        print("      → Existe asimetría en la dinámica del precio")
        print("      → Las subidas y bajadas no son simétricas")
    else:
        print("\n   ⚠️  La irreversibilidad no es concluyente")

    # ---------------------------------------------------------------
    # 6. ANÁLISIS FRACTAL
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ANÁLISIS FRACTAL — LEY DE POTENCIA")
    print("=" * 65)

    r, H, r2 = ajustar_ley_potencia(k_total_vals, P_total)
    print(f"\n   P(k) ~ k^(-r)")
    print(f"   Exponente r:           {r:.4f}")
    print(f"   R² del ajuste:         {r2:.4f}")
    print(f"   Exponente de Hurst H:  {H:.4f}   (H = (3-r)/2)")

    if H > 0.5:
        print(f"\n   📈 Serie PERSISTENTE (H = {H:.4f} > 0.5)")
        print("      → Las tendencias tienden a mantenerse")
        print("      → Existe memoria de largo plazo")
    elif H < 0.5:
        print(f"\n   📉 Serie ANTI-PERSISTENTE (H = {H:.4f} < 0.5)")
        print("      → Las tendencias tienden a revertirse")
    else:
        print("   ↔️  Sin correlación de largo alcance (H ≈ 0.5)")

    # ---------------------------------------------------------------
    # 7. VISUALIZACIONES
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Análisis de Visibility Graph — Apple (AAPL) 2020-2025',
                 fontsize=16, fontweight='bold', y=1.02)

    # --- 7a. Serie temporal ---
    ax = axes[0, 0]
    ax.plot(fechas, precios, 'steelblue', linewidth=0.8)
    ax.fill_between(fechas, precios, alpha=0.1, color='steelblue')
    ax.set_title('Precio de cierre diario', fontsize=13)
    ax.set_ylabel('Precio (USD)')
    ax.grid(True, alpha=0.3)

    # --- 7b. Distribuciones dirigidas (log-log) ---
    ax = axes[0, 1]
    ax.scatter(k_in_vals, P_in, c='royalblue', marker='o', s=50,
               alpha=0.7, edgecolors='navy', label=r'$P_{in}(k)$', zorder=5)
    ax.scatter(k_out_vals, P_out, c='crimson', marker='s', s=50,
               alpha=0.7, edgecolors='darkred', label=r'$P_{out}(k)$', zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Grado k')
    ax.set_ylabel('P(k)')
    ax.set_title(f'Distribuciones dirigidas\nKLD = {kld_original:.4f}', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    # --- 7c. Distribución total + ajuste ley de potencia ---
    ax = axes[0, 2]
    ax.scatter(k_total_vals, P_total, c='forestgreen', marker='o', s=50,
               alpha=0.7, edgecolors='darkgreen', label='P(k) empírica', zorder=5)
    # Línea de ajuste
    mascara = (k_total_vals >= 2) & (P_total > 0)
    if mascara.any():
        k_ajuste = k_total_vals[mascara].astype(float)
        p_pred = np.exp(np.log(P_total[mascara][0]) +
                        (-r) * (np.log(k_ajuste) - np.log(k_ajuste[0])))
        ax.plot(k_ajuste, p_pred, 'r--', linewidth=2.5,
                label=f'Ajuste: $k^{{-{r:.2f}}}$  (H={H:.2f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Grado k')
    ax.set_ylabel('P(k)')
    ax.set_title('Distribución no-direccional\n(Ley de potencia)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # --- 7d. Secuencia de grados ---
    ax = axes[1, 0]
    ax.plot(range(N), k_out, 'r-', linewidth=0.4, alpha=0.6, label=r'$k_{out}$')
    ax.plot(range(N), k_in, 'b-', linewidth=0.4, alpha=0.6, label=r'$k_{in}$')
    ax.set_xlabel('Nodo (orden temporal)')
    ax.set_ylabel('Grado')
    ax.set_title('Secuencias de grado dirigidas', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- 7e. Comparación P_in vs P_out (barras) ---
    ax = axes[1, 1]
    k_max_bar = min(40, max(k_in_vals.max(), k_out_vals.max()) + 1)
    rango_k = np.arange(0, k_max_bar + 1)
    pin_dict = dict(zip(k_in_vals, P_in))
    pout_dict = dict(zip(k_out_vals, P_out))
    pin_bar = [pin_dict.get(k, 0) for k in rango_k]
    pout_bar = [pout_dict.get(k, 0) for k in rango_k]
    w = 0.35
    ax.bar(rango_k - w/2, pin_bar, w, alpha=0.7, color='royalblue',
           label=r'$P_{in}$')
    ax.bar(rango_k + w/2, pout_bar, w, alpha=0.7, color='crimson',
           label=r'$P_{out}$')
    ax.set_xlabel('Grado k')
    ax.set_ylabel('Probabilidad')
    ax.set_title(f'$P_{{in}}$ vs $P_{{out}}$  (asimetría temporal)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.5, k_max_bar + 0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # --- 7f. Visualización del VG (subconjunto) ---
    ax = axes[1, 2]
    n_vis = min(50, N)
    precios_vis = precios[:n_vis]
    ax.bar(range(n_vis), precios_vis, color='steelblue', alpha=0.6, width=0.8)

    # Dibujar aristas de visibilidad
    aristas_vis = [(i, j) for i, j in aristas if i < n_vis and j < n_vis]
    for i, j in aristas_vis:
        ax.plot([i, j], [precios_vis[i], precios_vis[j]],
                'r-', alpha=0.12, linewidth=0.6)

    ax.set_xlabel('Tiempo (índice)')
    ax.set_ylabel('Precio')
    ax.set_title(f'Visibility Graph visual\n(primeros {n_vis} datos, '
                 f'{len(aristas_vis)} aristas)', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('apple_visibility_graph.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---------------------------------------------------------------
    # 8. RESUMEN FINAL
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESUMEN COMPLETO")
    print("=" * 65)
    print(f"""
    ┌─────────────────────────────────────────────────────┐
    │  Serie:  Apple (AAPL) — Precio de cierre diario     │
    │  Período: 2020-01-01 → 2025-01-01 ({N} datos)      │
    ├─────────────────────────────────────────────────────┤
    │  VISIBILITY GRAPH                                    │
    │    Aristas:       {len(aristas):>8}                          │
    │    Grado medio:   {k_total.mean():>8.2f}                          │
    │    Grado máximo:  {k_total.max():>8}                          │
    ├─────────────────────────────────────────────────────┤
    │  IRREVERSIBILIDAD TEMPORAL                           │
    │    KLD original:   {kld_original:>10.6f}                      │
    │    KLD reshuffled: {kld_ref:>10.6f}                      │
    │    Ratio:          {kld_original/kld_ref:>10.1f}x                     │
    ├─────────────────────────────────────────────────────┤
    │  ANÁLISIS FRACTAL                                    │
    │    Exponente r:    {r:>10.4f}                          │
    │    Hurst H:        {H:>10.4f}                          │
    │    R² ajuste:      {r2:>10.4f}                          │
    └─────────────────────────────────────────────────────┘
    """)

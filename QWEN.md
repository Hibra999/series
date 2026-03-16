# Visibility Graph Analysis Project

## Project Overview

This project implements **Visibility Graph (VG) analysis** for time series data, specifically applied to Apple (AAPL) stock closing prices from 2020-2025. The implementation is based on the research paper:

> Xiong, H., Shang, P., Xia, J., Wang, Y. (2018). "Time irreversibility and intrinsics revealing of series with complex network approach", *Physica A*.

### Purpose

The project converts time series data into complex networks (visibility graphs) to analyze:
- **Time Irreversibility**: Whether the series behaves differently forward vs backward in time
- **Fractal Properties**: Long-range correlations and self-similarity via power-law analysis
- **Hurst Exponent**: Persistence/anti-persistence behavior in financial data

### Key Algorithms

| Algorithm | Complexity | Description |
|-----------|------------|-------------|
| Visibility Graph Construction | O(N²) | Optimized algorithm using slope-based visibility checks |
| Brute-force VG Verification | O(N³) | Reference implementation for validation |
| Kullback-Leibler Divergence | O(N) | Measures asymmetry between in/out degree distributions |
| Power-law Fitting | O(N) | Log-log regression for fractal analysis |

## Technologies

- **Language**: Python 3.x
- **Core Libraries**:
  - `numpy` - Numerical computations
  - `pandas` - Data handling
  - `matplotlib` - Visualization
  - `scipy.stats` - Statistical analysis (linear regression)
- **Optional**:
  - `yfinance` - Yahoo Finance API for real stock data

## Installation

```bash
# Required dependencies
pip install numpy pandas matplotlib scipy

# Optional: for real stock data
pip install yfinance
```

## Usage

### Running the Analysis

```bash
python visibility_graph_analysis.py
```

The script will:
1. Download AAPL data (or generate synthetic data if yfinance unavailable)
2. Build the visibility graph
3. Calculate degree distributions (in/out/total)
4. Compute KLD for time irreversibility analysis
5. Fit power-law and estimate Hurst exponent
6. Generate visualization saved as `apple_visibility_graph.png`

### Customization

Modify the parameters in the `__main__` section:

```python
# Change stock ticker or date range
precios, fechas = descargar_datos("AAPL", "2020-01-01", "2025-01-01")

# Or use custom time series
serie_personalizada = tu_array_de_datos
```

## Output Metrics

### Irreversibility Analysis
- **KLD ≈ 0**: Time-reversible series (symmetric dynamics)
- **KLD >> 0**: Time-irreversible series (asymmetric up/down movements)

### Fractal Analysis
- **H > 0.5**: Persistent series (trends tend to continue)
- **H < 0.5**: Anti-persistent series (trends tend to reverse)
- **H ≈ 0.5**: Random walk / no long-range correlation

## Project Structure

```
series/
├── visibility_graph_analysis.py   # Main analysis script
├── QWEN.md                        # Project documentation
├── apple_visibility_graph.png     # Generated visualization (after running)
└── .git/                          # Git repository
```

## Key Functions

| Function | Description |
|----------|-------------|
| `descargar_datos()` | Downloads stock data from Yahoo Finance |
| `construir_visibility_graph()` | Builds VG with optimized O(N²) algorithm |
| `vg_fuerza_bruta()` | Brute-force VG for verification |
| `distribucion_grado()` | Computes degree distribution P(k) |
| `calcular_kld()` | Calculates Kullback-Leibler divergence |
| `ajustar_ley_potencia()` | Power-law fit and Hurst estimation |

## References

- **Paper**: Xiong et al. (2018), Physica A - Time irreversibility analysis via visibility graphs
- **Visibility Graph**: Lacasa et al. (2008) - Original VG formulation for time series

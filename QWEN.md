# Series — Visibility Graph Analysis for Financial Time Series

## Project Overview

This project implements **Visibility Graph (VG) analysis** for financial time series data, specifically applied to Apple (AAPL) stock prices from 2020-2025. The Visibility Graph method transforms time series data into complex networks, enabling the analysis of:

- **Temporal irreversibility** — Detecting asymmetry between past and future through Kullback-Leibler Divergence (KLD)
- **Fractal properties** — Estimating Hurst exponent via power-law degree distribution fitting
- **Market predictability** — Using VG features for price direction classification

## Technologies & Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `numba` | JIT compilation for performance-critical VG algorithms |
| `scikit-learn` | Gradient Boosting classifier for prediction |
| `yfinance` | Stock data download |
| `scipy` | Statistical analysis (power-law fitting) |

## Project Structure

```
series/
├── v1.py                        # Full VG-based forecasting system (VGFS)
├── visibility_graph_analysis.py # Detailed VG analysis with validation
├── *.png                        # Generated analysis visualizations
└── QWEN.md                      # This documentation
```

## Key Files

### `v1.py` — Visibility Graph Forecasting System (VGFS)

A complete ML pipeline that:
- Downloads AAPL stock data via yfinance
- Extracts 19 features: 12 VG-based + 7 traditional technical indicators
- Trains Gradient Boosting classifiers on different feature subsets
- Generates 5 visualization dashboards

**Key functions:**
- `vg(s)` — Computes in-degree, out-degree, and total degree sequences
- `kld(ki, ko)` — Kullback-Leibler divergence between in/out distributions
- `hs(kt)` — Hurst exponent estimator from degree distribution
- `fvg(s)` — 12-dimensional VG feature vector
- `ftr(s)` — 7-dimensional traditional technical features
- `bds(p, v, h)` — Builds dataset with sliding windows

### `visibility_graph_analysis.py` — Analytical Validation

Provides detailed analysis with:
- Brute-force VG verification for correctness
- Shuffle tests for irreversibility significance
- Power-law exponent estimation with R² goodness-of-fit
- Comprehensive 6-panel visualization

## Building and Running

### Prerequisites

```bash
pip install numpy pandas matplotlib numba scikit-learn yfinance scipy
```

### Run Full Forecasting System

```bash
python v1.py
```

**Outputs:**
- `AAPL_dash.png` — Dashboard with price, cumulative returns, Hurst exponent, confusion matrix
- `AAPL_fvr.png` — Feature comparison: bar chart, ROC curve, P-R curve, returns
- `AAPL_vg.png` — VG properties: degree distribution, degree sequence, scatter plot
- `AAPL_rol.png` — Rolling analysis: price, probabilities, returns, cumulative accuracy
- `AAPL_feat.png` — Feature analysis: correlation bars, correlation heatmap

### Run Detailed Analysis

```bash
python visibility_graph_analysis.py
```

**Output:**
- `apple_visibility_graph.png` — 6-panel comprehensive analysis figure

## Algorithm Details

### Visibility Graph Construction

Two points `(i, s[i])` and `(j, s[j])` are connected if all intermediate points lie below the visibility line:

```
s[k] < s[i] + (s[j] - s[i]) × (k - i) / (j - i)  for all k ∈ (i, j)
```

### Feature Extraction

**VG Features (12):**
1. Last in-degree
2. Normalized last in-degree
3. KLD (irreversibility)
4. Hurst exponent
5. Mean degree
6. Skewness of degree distribution
7. Growth ratio (recent vs early)
8. Slope of recent in-degrees
9. Maximum degree
10. Coefficient of variation
11. Recent out-in difference
12. Entropy of degree distribution

**Technical Features (7):**
1. Total return
2. Volatility
3. Return skewness
4. Return kurtosis
5. Drawdown ratio
6. Relative position in range
7. Recent momentum

## Development Conventions

- **Code style:** Compact, performance-oriented (minified imports, single-letter variables in numba functions)
- **JIT compilation:** All performance-critical loops use `@njit` with `fastmath` and `prange` for parallelization
- **Visualization:** Dark theme (`dark_background` style) for all plots
- **Random state:** Fixed at 42 for reproducibility

## Key Metrics

| Metric | Interpretation |
|--------|----------------|
| KLD > 3× shuffle | Strong temporal irreversibility |
| H > 0.5 | Persistent (trend-following) behavior |
| H < 0.5 | Anti-persistent (mean-reverting) behavior |
| H ≈ 0.5 | Random walk |

## Output Interpretation

The forecasting system compares three models:
- **VG** — Visibility Graph features only
- **TR** — Traditional technical indicators only
- **ALL** — Combined feature set

Best model is selected by accuracy and used for trading signal generation, compared against Buy & Hold baseline.

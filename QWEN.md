# Series - Stock Price Prediction with Adaptive Stacking Ensembles

## Project Overview

This is a quantitative finance machine learning project for **stock price direction classification and price forecasting**. The project implements progressively sophisticated ensemble models using gradient boosting (CatBoost, LightGBM, XGBoost) combined with neural network meta-learners (TimesNet) and adaptive walk-forward validation.

### Key Features

- **Custom Mathematical Feature Engineering**: Numba-optimized functions for volatility graph analysis (`vg`), KL divergence (`kld`), and Hurst exponent estimation (`hs`)
- **19-Dimensional Feature Space**: 12 volatility graph features + 7 traditional technical indicators
- **Stacking Ensemble**: 3-level architecture with base learners (CatBoost/LightGBM/XGBoost) and meta-learners (Logistic Regression/Ridge/TimesNet)
- **Target Transformations**: 6 bijective transformations for price forecasting (PIT+Logit, Normalizing Flow, Rational Spline, Yeo-Johnson, Log+StandardScaler, Sinh-Arcsinh)
- **Multi-Horizon Prediction**: Simultaneous forecasting at horizons [1, 3, 5, 10] days
- **Adaptive Walk-Forward Validation**: Error-based retraining triggered when prediction error exceeds baseline threshold
- **Multi-Threaded Training**: CPU-optimized with `n_jobs=-1` and `thread_count=-1`

## Project Structure

```
series/
├── v1.py              # Base version: Simple stacking with walk-forward
├── v2.py              # Extended: Multiple target transformations comparison
├── v3.py              # Advanced: Multi-horizon + TimesNet meta-learner + adaptive retraining
├── run_output.log     # Latest execution log
├── catboost_info/     # CatBoost training metrics and logs
└── __pycache__/       # Compiled Python bytecode
```

## Version Evolution

| Version | Key Additions |
|---------|---------------|
| `v1.py` | Base stacking ensemble, Ridge/LogReg meta-learners, single horizon (h=5) |
| `v2.py` | 6 target transformations, HTML report generation, transformation comparison |
| `v3.py` | Multi-horizon features, TimesNet meta-learner, error-based adaptive retraining, stride-based walk-forward |

## Dependencies

The project requires the following Python packages:

```
numpy, pandas, yfinance, matplotlib
numba (JIT compilation)
scikit-learn (preprocessing, metrics)
catboost, lightgbm, xgboost (gradient boosting)
torch (PyTorch for TimesNet and Normalizing Flow)
scipy (interpolation, transformations)
```

## Running the Models

```bash
# Run base version (v1)
python v1.py

# Run transformation comparison (v2) - generates HTML report
python v2.py

# Run multi-horizon with TimesNet (v3) - most advanced
python v3.py
```

### Configuration Parameters

Key parameters in `VGSS` class initialization:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tk` | "AAPL" | Stock ticker symbol |
| `v` | 40 | Lookback window for feature calculation |
| `h` | 5 | Prediction horizon (days) |
| `horizons` | [1,3,5,10] | Multi-horizon prediction points (v3 only) |
| `rt` | 0.7 | Train/test split ratio |
| `err_window` | 10 | Error tracking window size (v3 only) |
| `err_thresh` | 1.5 | Error ratio threshold for retraining (v3 only) |

## Output Files

- **`{ticker}_{transform}_adaptive.png`**: Visualization plots for price and direction predictions
- **`metricas_transformaciones.html`**: Comprehensive HTML report comparing all transformations
- **`catboost_info/`**: CatBoost training logs and metrics

## Architecture Details

### Feature Engineering (`fvg`, `ftr`)

```
12 Volatility Graph Features:
  - ki[-1], ki[-1]/max(kt), kld(ki,ko), hs(kt)
  - Mean, skew, growth ratio, slope, max, std/mean
  - Recent momentum, entropy

7 Traditional Features:
  - Return, volatility, skewness, kurtosis
  - Drawdown ratio, relative position, MA deviation
```

### Stacking Pipeline

```
Level 0 (Base): CatBoost, LightGBM, XGBoost (classification + regression)
       ↓
Level 1 (Meta): Logistic Regression (direction) + Ridge/TimesNet (price)
       ↓
Output: Direction probability + Price forecast (inverse-transformed)
```

### Adaptive Retraining Logic (v3)

Retraining is triggered when:
1. Median recent error / baseline error > `err_thresh` (1.5)
2. OR every 50 steps (periodic refresh)

## Development Practices

- **Numba JIT**: All feature computation functions use `@njit(fastmath=True, cache=True)` for performance
- **Numerical Stability**: Epsilon clipping, bounded transformations, gradient clipping in PyTorch
- **Memory Efficiency**: `np.ascontiguousarray` for yfinance data, in-place operations where possible
- **Reproducibility**: Fixed random seeds (42) across all models

## Typical Results

Based on `run_output.log` (AAPL, 2015-2025 train + 2026 test):

| Transformation | Direction Accuracy | Price R² |
|----------------|-------------------|----------|
| Sktime_Pipeline_LogSS | 0.5481 | **0.7791** |
| Normalizing_Flow_PyTorch | 0.5481 | 0.7740 |
| Rational_Spline_PCHIP | 0.5481 | 0.7282 |
| INN_Yeo_Johnson | 0.5481 | 0.7278 |
| PIT_Aprendida_Logit | 0.5481 | 0.6877 |
| Omni_Custom_SinhArcsinh | 0.5481 | 0.5069 |

## Notes

- The project downloads historical data from Yahoo Finance automatically
- Training data: 2015-01-01 to 2025-12-31
- Test data: 2026-01-01 onwards (forward validation)
- All visualizations are saved as high-DPI PNG files (150 DPI)

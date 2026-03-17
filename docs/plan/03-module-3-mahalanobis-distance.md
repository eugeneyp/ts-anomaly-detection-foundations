## Module 3: Mahalanobis distance — multivariate baseline detection

**Dataset:** SKAB (train on `anomaly-free.csv`, test on `valve1/`, `valve2/`, `other/` experiments) and SWaT (train on `normal.csv`, test on `merged.csv`)

**What you'll build.** The leap from univariate to multivariate. Compute Mahalanobis distance from the healthy baseline for each timestep across all sensor channels simultaneously. This captures cross-sensor correlations that univariate z-scores miss entirely.

---

## What was actually implemented

### Notebooks

| Notebook | Dataset | Description |
|----------|---------|-------------|
| `03-skab-valve1-0.ipynb` | SKAB valve1/0.csv | Initial exploration: single experiment, Mahalanobis vs z-score |
| `03-skab-valve1-benchmark.ipynb` | SKAB valve1 (16 experiments) | Benchmark loop across all 16 valve1 experiments; per-dataset F1 table + bar chart |
| `03-skab-valve2-benchmark.ipynb` | SKAB valve2 (16 experiments) | Same benchmark on valve2 experiments |
| `03-skab-other-benchmark.ipynb` | SKAB other (rotor imbalance, leaks, cavitation) | Same benchmark on other anomaly types |
| `03-swat-benchmark.ipynb` | SWaT (1.44M rows, 26 sensors) | Full benchmark on SWaT using `merged.csv`; decisive Mahalanobis advantage demonstrated |

### Implementation

**Core helper functions** (shared across all benchmark notebooks):

```python
def compute_zscore_predictions(train, test, features, threshold=3.0):
    mu = train[features].mean()
    std = train[features].std()
    z_scores = (test[features] - mu) / std
    anomaly_per_sensor = (np.abs(z_scores) > threshold).astype(int)
    return anomaly_per_sensor.max(axis=1)  # flag if ANY sensor exceeds threshold

def compute_mahal_predictions(train, test, features, chi2_threshold):
    mu = train[features].mean().values
    cov = train[features].cov().values
    cov_inv = np.linalg.pinv(cov)  # pseudo-inverse for numerical stability
    diff = test[features].values - mu
    t2_scores = (diff * (diff @ cov_inv)).sum(axis=1)
    return (t2_scores > chi2_threshold).astype(int)
```

**Threshold:** Chi-squared 99.9th percentile with `df = len(features)` degrees of freedom.

**Z-score threshold:** 3σ (flag if any single sensor exceeds).

**SKAB setup:** Train on first 400 rows of each file as the normal baseline; evaluate on the full file.

**SWaT setup:** Train on all of `normal.csv` (1,387,098 rows); evaluate on `merged.csv` (1,441,719 rows, time-sorted). Test set includes both normal (96.2%) and attack (3.8%) rows.

**SWaT feature selection:** 26 continuous sensors only (exclude 25 binary actuator columns — P### pumps and MV### motorized valves — which produce a near-singular covariance matrix).

---

## Key findings

### SKAB (valve1, valve2, other)

**Both methods perform similarly** — typically within ~0.01 F1 of each other across all 48 experiments. Neither method consistently wins.

**Why:** SKAB anomalies are predominantly large-magnitude single-sensor events (valve closure → obvious pressure spike, rotor imbalance → obvious vibration spike). When anomalies are "univariate" in character, z-score and Mahalanobis produce nearly identical predictions.

### SWaT (26 sensors, 1.44M rows)

| Method | F1 | Precision | Recall |
|--------|----|-----------|--------|
| Univariate Z-Score | 0.4249 | 0.2967 | 0.7480 |
| **Mahalanobis Distance** | **0.7240** | **0.7277** | **0.7202** |

**Mahalanobis wins decisively** (+0.30 F1). Both methods achieve similar recall (~72–75%), but z-score generates **6.6× more false positives** on normal data (96,858 vs 14,718).

**Why z-score over-fires:** With 26 sensors, the union-bound false-alarm rate is approximately `1 - (1 - 0.0027)^26 ≈ 6.8%` per timestep. Over 1.38M normal rows this produces ~96K false alarms.

**Why Mahalanobis stays quiet:** The inverse covariance matrix encodes the full correlation structure learned from 1.38M training rows. Correlated sensor movements during normal operation are down-weighted; only joint deviations that violate the learned correlation pattern exceed the chi-squared threshold.

---

## Data notes (SWaT-specific)

- **Separator:** comma
- **Column names have leading spaces** — must call `df.columns = df.columns.str.strip()`
- **Label column:** `Normal/Attack` — map `"Attack" → 1, "Normal" → 0`
- **Timestamp format:** `DD/MM/YYYY H:MM:SS AM/PM` (single-digit hour, no zero-pad) — parse with `format='%d/%m/%Y %I:%M:%S %p'`
- **`merged.csv` is not time-sorted** — it is `normal.csv` rows concatenated with `attack.csv` rows. Must call `.sort_index()` after setting the datetime index for correct time-series visualization.
- **Use `merged.csv` for evaluation** (not `attack.csv`). `attack.csv` contains only attack rows (100% attack rate), making precision trivially 1.0 and F1 a recall-only metric.

---

## Practical takeaway

| Condition | Prefer |
|-----------|--------|
| Few sensors, large-magnitude anomalies | Univariate Z-Score — simpler, fewer parameters |
| Many correlated sensors, stealthy anomalies | **Mahalanobis Distance** — encodes joint structure, far fewer false alarms |
| Small training set (< few thousand rows) | Univariate Z-Score — covariance matrix estimate is unstable |
| Mixed binary + continuous features | Univariate Z-Score — binary columns make covariance near-singular |
| Need per-sensor interpretability | Univariate Z-Score — Mahalanobis gives one distance score, not per-sensor |

---

## Original implementation plan (for reference)

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

features = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
            'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'RateRMS']
train = pd.read_csv('anomaly-free/anomaly-free.csv')[features]
mu = train.mean().values
cov = train.cov().values
cov_inv = np.linalg.pinv(cov)

def mahal_score(row):
    return mahalanobis(row, mu, cov_inv)

test = pd.read_csv('valve1/1.csv')
test['mahal_dist'] = test[features].apply(mahal_score, axis=1)

p = len(features)
threshold_99 = chi2.ppf(0.99, df=p)
test['anomaly'] = (test['mahal_dist'] ** 2 > threshold_99).astype(int)
```

**Production connection.** Hotelling's T² on multivariate sensor data is the statistical backbone of GE SmartSignal's digital twin residual approach and is implemented in the MathWorks Predictive Maintenance Toolbox. Mahalanobis distance on engineered vibration features is what many production anomaly detectors compute under the hood.

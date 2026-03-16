## Module 3: Mahalanobis distance — multivariate baseline detection

**Dataset:** SKAB (train on `anomaly-free.csv`, test on `valve1/` and `other/` experiments)

**What you'll build.** The leap from univariate to multivariate. Compute Mahalanobis distance from the healthy baseline for each timestep across all 8 SKAB sensor channels simultaneously. This captures cross-sensor correlations that univariate z-scores miss entirely.

**Implementation.**

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# Train: compute mean and covariance from anomaly-free data
features = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
            'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'RateRMS']
train = pd.read_csv('anomaly-free/anomaly-free.csv')[features]
mu = train.mean().values
cov = train.cov().values
cov_inv = np.linalg.pinv(cov)  # use pseudo-inverse for numerical stability

# Score: Mahalanobis distance for each test point
def mahal_score(row):
    return mahalanobis(row, mu, cov_inv)

test = pd.read_csv('valve1/1.csv')
test['mahal_dist'] = test[features].apply(mahal_score, axis=1)

# Threshold: chi-squared distribution with p degrees of freedom
p = len(features)
threshold_95 = chi2.ppf(0.95, df=p)
threshold_99 = chi2.ppf(0.99, df=p)
test['anomaly'] = (test['mahal_dist'] ** 2 > threshold_99).astype(int)
```

**Critical experiment.** Run the same anomaly detection two ways: (a) univariate z-scores on each sensor independently, (b) multivariate Mahalanobis distance. For the valve closure experiments, a valve closing increases pressure while decreasing flow — neither individual sensor might cross its threshold, but the *combination* is highly anomalous. Mahalanobis catches this; independent z-scores likely miss it. Quantify the difference in F1 scores.

**Extend with Hotelling's T².** Mahalanobis distance squared is equivalent to Hotelling's T² statistic. Implement the T² control chart with upper control limit (UCL) derived from the F-distribution for formal statistical testing. This gives you the theoretically grounded threshold without arbitrary percentile choices.

**What you'll learn.** The power of multivariate methods comes from modeling feature correlations. When pressure and flow are positively correlated during normal operation, a simultaneous increase in pressure and decrease in flow is far more anomalous than either change alone. This is the mathematical reason ML reduces false alarms — it captures the multivariate "shape" of normal, not just the univariate bounds.

**Production connection.** Hotelling's T² on multivariate sensor data is the statistical backbone of GE SmartSignal's digital twin residual approach and is implemented in the MathWorks Predictive Maintenance Toolbox. Mahalanobis distance on engineered vibration features is what many production anomaly detectors compute under the hood.
## Module 4: Isolation Forest and One-Class SVM — classical ML

**Status: Completed**

**Notebooks:**
- `notebooks/04-swat-isolation-forest.ipynb` — IF on SWaT (51 features: 26 continuous + 25 binary actuators) vs. Mahalanobis
- `notebooks/04-skab-isolation-forest.ipynb` — IF on SKAB valve1/valve2/other vs. Mahalanobis

**OCSVM omitted.** SWaT has 1.38M training rows; OCSVM kernel matrix computation is O(n²) in memory (~7.6 TB at that scale). SKAB's 400-row training sets would be feasible, but the module was scoped to the more informative SWaT + SKAB IF comparison given the scale constraint is the key production lesson. The O(n²) constraint is documented in both notebooks.

---

### What was implemented

**SWaT (`04-swat-isolation-forest.ipynb`):**
- IF trained on 1.38M normal rows using all 51 features (26 continuous sensors + 25 binary actuators)
- `contamination=0.0379` (known test attack rate), `n_estimators=200`
- Contamination sweep (0.005 → 0.10), anomaly score distribution plot, time-series visualization
- Compared against Module 3 Mahalanobis (26 continuous features only)

**SKAB (`04-skab-isolation-forest.ipynb`):**
- IF trained on 400 normal rows per file, evaluated on full file (same split as Module 3)
- `contamination='auto'` (Liu et al. 2008 theoretical threshold, no anomaly rate estimate required)
- Benchmarked across valve1 (16 exp), valve2 (4 exp), other (14 exp)
- Compared against Mahalanobis recomputed in the same loop

---

### Results

**SWaT:**

| Method | Features | F1 | Precision | Recall |
|--------|----------|----|-----------|--------|
| Univariate Z-Score | 26 | 0.4249 | 0.2967 | 0.7480 |
| Mahalanobis Distance | 26 | **0.7240** | **0.7277** | **0.7202** |
| Isolation Forest | 51 | 0.5154 | 0.4145 | 0.6812 |

**SKAB (`contamination='auto'`):**

| Dataset | Mahal F1 | IF F1 | ΔF1 |
|---------|----------|-------|-----|
| Valve1 (16 exp) | **0.7435** | 0.6814 | −0.062 |
| Valve2 (4 exp)  | **0.6955** | 0.6872 | −0.008 |
| Other  (14 exp) | **0.7104** | 0.6634 | −0.047 |

Mahalanobis wins every benchmark. On SKAB, IF has higher recall (~0.87) but lower precision (~0.56); Mahalanobis achieves better balance.

---

### What was learned

**1. Model structure matching the anomaly type is the dominant factor.**
SWaT attacks are designed as stealthy covariance violations — exactly the regime Mahalanobis is built for. With 1.38M training rows providing a stable covariance estimate, Mahalanobis's explicit encoding of all 351 pairwise sensor relationships dominates IF's randomized feature selection (+0.21 F1).

**2. Binary actuators did not help IF on SWaT.**
The hypothesis that including 25 binary actuator columns would give IF an advantage did not hold. SWaT's 36 attack scenarios predominantly spoof sensor readings while leaving actuator states unchanged — the anomaly signal is in sensor-to-sensor correlations, not sensor-to-actuator relationships. The extra columns added noise without adding signal.

**3. IF's threshold calibration is a structural weakness.**
Mahalanobis uses a chi-squared critical value that self-calibrates to the learned covariance structure — no anomaly rate estimate required. IF's `contamination` parameter requires an empirical prior:
- `contamination=0.05` on SKAB (true rate ~20–40%): threshold too conservative, recall collapses to 0.58 on Valve1
- `contamination='auto'` (Liu et al. theoretical midpoint): threshold too permissive, precision drops to 0.56
- `contamination='auto'` is strictly better than fixed values when the anomaly rate is unknown, but still doesn't match Mahalanobis's naturally balanced P/R

**4. The prediction about `other` experiments did not materialise.**
The module plan predicted IF would outperform Mahalanobis on cavitation, rotor imbalance, and fluid leaks (nonlinear anomalies). It did not — the threshold calibration penalty offset any scoring advantage from IF's non-parametric nature.

**5. OCSVM scale constraint is a real production blocker.**
At 1.38M rows, the RBF kernel matrix alone would be ~7.6 TB. This is not a tuning problem — it's a fundamental algorithmic constraint. IF's sublinear training time makes it the practical default for large industrial datasets.

---

### What you'll learn (original, confirmed by results)

Isolation Forest's key insight — anomalies are *easy to isolate*, not necessarily *far from normal* — is a genuinely different philosophy from distance-based methods. But the results show this philosophical difference doesn't automatically translate to better performance when the anomaly type (covariance violations, large spikes) is well-handled by simpler methods with principled thresholds.

**Production connection.** Isolation Forest remains the most deployed classical ML anomaly detector on edge hardware (ESP32 microcontrollers, STMicroelectronics NanoEdge AI Studio) because of its low memory footprint and fast inference. The threshold calibration challenge is real in production — `contamination='auto'` is the right default when attack rates are unknown.

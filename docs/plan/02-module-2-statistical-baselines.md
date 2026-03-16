## Module 2: Statistical baselines — moving statistics and z-scores

**Dataset:** NAB (all three files)

**What you'll build.** Replace the fixed threshold with an adaptive one: compute a rolling mean and rolling standard deviation, then flag points that deviate by more than k standard deviations. This is the z-score approach and the foundation of statistical process control.

**Implementation.**

```python
# Rolling z-score
window = 168  # e.g., 1 week of hourly data
df['rolling_mean'] = df['value'].rolling(window).mean()
df['rolling_std'] = df['value'].rolling(window).std()
df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
df['anomaly_zscore'] = (df['z_score'].abs() > 3).astype(int)
```

**Extend to EWMA and CUSUM.** Implement exponentially weighted moving average (EWMA) control chart — `df['ewma'] = df['value'].ewm(span=window).mean()` — which weights recent observations more heavily. Then implement a CUSUM (Cumulative Sum) chart that accumulates small deviations over time:

```python
# CUSUM: detects persistent small shifts
target = df['value'][:baseline_end].mean()
k = 0.5 * df['value'][:baseline_end].std()  # slack parameter
h = 5 * df['value'][:baseline_end].std()     # threshold
cusum_pos, cusum_neg = [0], [0]
for v in df['value']:
    cusum_pos.append(max(0, cusum_pos[-1] + (v - target) - k))
    cusum_neg.append(max(0, cusum_neg[-1] - (v - target) - k))
df['cusum_alarm'] = [(p > h or n > h) for p, n in zip(cusum_pos[1:], cusum_neg[1:])]
```

**Key comparison.** Run z-score, EWMA, and CUSUM on the same three NAB files. You'll discover that z-score catches sudden spikes well but misses gradual drift; CUSUM catches gradual drift beautifully but over-alarms on volatile signals; EWMA is a good compromise. Plot all three anomaly scores on the same timeline with the ground truth.

**What you'll learn.** The window size (lookback period) is the single most important hyperparameter — it determines what counts as "normal." A 1-hour window adapts quickly but false-alarms on normal daily patterns. A 1-week window captures daily cycles but responds slowly to real faults. This tradeoff between sensitivity and stability is the fundamental design tension in every PdM system.

**Theory.** The z-score assumes the signal follows a Gaussian distribution within the rolling window. When that assumption holds (e.g., steady-state machine operation), this works well. When it doesn't (e.g., startup transients, bimodal speed distributions), it breaks. CUSUM's power comes from accumulation — a +0.3σ shift sustained over 20 samples adds up to a 6σ cumulative deviation even though no individual sample is alarming. This is why Page (1954) designed it for industrial quality control.

**Production connection.** The z-score with rolling windows is essentially what GE SmartSignal computes — the residual between predicted and actual, normalized by the expected variation at those operating conditions. CUSUM charts are standard in pharmaceutical and semiconductor manufacturing SPC.

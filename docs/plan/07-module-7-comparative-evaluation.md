## Module 8: Comparative evaluation and synthesis

**Dataset:** All datasets, with final focus on SKAB

**Status: Not started**

**What you'll build.** A comprehensive comparison framework that answers the question you asked: *how much does each technique actually improve over the simpler one?*

**The comparison table you'll produce:**

For each SKAB anomaly experiment, compute:

- Detection lag (how many timesteps after the anomaly starts before the model alarms)
- False alarm rate (alarms in the normal portion)
- F1 score
- Computational cost (training time, inference time per sample)

For these methods:

1. Fixed threshold (best single-feature threshold)
2. Rolling z-score (best window size)
3. CUSUM
4. Mahalanobis distance
5. Isolation Forest
6. Random Cut Forest
7. Standard Autoencoder
8. LSTM-Autoencoder

**Visualization.** Create a heatmap where rows are methods, columns are SKAB anomaly types (valve1, valve2, rotor imbalance, fluid leak, cavitation), and cell color represents F1 score. This single visualization will crystallize the key insight: simple methods work well on *some* anomaly types (especially sudden process changes), while ML methods earn their keep on *others* (gradual drift, subtle multivariate shifts).

**Health indicator construction.** Take the anomaly scores from your best-performing models and transform them into smooth, monotonic health indicators using the EWMA + isotonic regression pipeline from the previous project guide. Compare the health indicator quality (monotonicity, trendability) across methods.

**NASA SMAP/MSL column.** For the deep learning methods (Autoencoder, LSTM-AE), include a second comparison column using the NASA benchmark. This surfaces the dataset-dependent nature of performance — methods that dominate on SKAB (short experiments, abrupt anomalies) may rank differently on NASA (long continuous telemetry, gradual drift).

**Write-up.** Produce a 1-page summary answering: "If I could only deploy one method, which would it be and why?" The answer will depend on constraints — for edge deployment with <1KB RAM, Isolation Forest wins. For streaming data with concept drift, RCF wins. For cloud deployment on critical assets, LSTM-AE wins. For cold-start with no training data, Mahalanobis with fleet-average baselines wins. This is the PM judgment the project builds.

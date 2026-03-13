# Time series anomaly detection: a foundations-first learning project

**Build domain-agnostic intuition for anomaly detection before specializing in vibration.** This project progresses from the simplest possible baseline (fixed threshold) through statistical methods, classical ML, and deep learning — all on the same datasets, so you see exactly what each layer of sophistication adds. Every technique maps to a production deployment pattern you'll encounter at AssetWatch.

---

## Why this sequencing matters

The vibration-specific projects in your research brief involve two entangled problems: understanding the signal processing physics (FFT, envelope analysis, bearing defect frequencies) and understanding the anomaly detection methodology (reconstruction error, threshold selection, health indicators). Learning both simultaneously means you can't tell whether a model fails because of a bad architecture choice or because you misprocessed the signal. This project isolates the anomaly detection problem on clean, well-labeled industrial time series — so when you later apply these same techniques to vibration data, you'll know exactly what the model is doing and why.

---

## Dataset selection: three datasets, increasing complexity

You'll use three datasets across the project. Each introduces a new challenge dimension.

### Dataset 1: Numenta Anomaly Benchmark (NAB) — univariate, labeled, diverse

**What it is.** 58 univariate time series from real-world sources: AWS CloudWatch server metrics, online ad click rates, NYC taxi demand, and machine temperature sensors. Each series has expert-labeled anomaly windows. This is the field's most widely used introductory benchmark.

**Why start here.** Univariate data strips away the complexity of multivariate correlation — you're detecting anomalies in a single stream. The diverse domains (server load, traffic, temperature) let you see how the same algorithm performs on different signal characteristics. The labeled anomaly windows enable proper precision/recall evaluation from day one.

**Download.** https://github.com/numenta/NAB — the `data/` folder contains all time series as CSVs; `labels/combined_windows.json` has the anomaly labels.

**Start with these specific files:**
- `realKnownCause/machine_temperature_system_failure.csv` — closest to industrial PdM, shows temperature drift before a machine failure
- `realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv` — server CPU with sudden spikes (point anomalies)
- `realTraffic/speed_7578.csv` — periodic traffic data with contextual anomalies (unusual patterns at normal-looking values)

### Dataset 2: SKAB (Skoltech Anomaly Benchmark) — multivariate, industrial, labeled

**What it is.** 35 experiments from a real water circulation testbed at Skoltech (Moscow) with a motor-driven pump, valves, and 8 sensor channels: two accelerometers (vibration), motor current, pressure, temperature, thermocouple, voltage, and flow rate. Each experiment contains one injected anomaly — valve closures, fluid leaks, rotor imbalance, cavitation. Every point is labeled as normal/anomalous.

**Why use this.** It's multivariate industrial sensor data from actual equipment — the closest publicly available analog to what AssetWatch sensors capture. You'll see how anomalies in one sensor propagate to others (closing a valve changes pressure, which changes flow, which changes vibration). The diversity of anomaly types (valve faults, imbalance, cavitation, leaks) maps directly to real maintenance scenarios. The fact that it includes both vibration and process variables bridges your foundations work into the vibration domain naturally.

**Download.** https://github.com/waico/SKAB or https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab

**Key files:**
- `anomaly-free/anomaly-free.csv` — your training baseline (normal operation)
- `valve1/` — valve closure experiments (sudden process change)
- `valve2/` — different valve closure experiments
- `other/` — rotor imbalance, fluid leaks, cavitation (gradual anomalies)

### Dataset 3: SWaT (Secure Water Treatment) — large-scale multivariate, cyber-physical

**What it is.** 946,719 timesteps from a real water treatment plant testbed at Singapore University of Technology and Design. 51 sensor channels across 6 process stages. 7 days of normal operation followed by 4 days of 36 different attack scenarios — valve manipulations, sensor spoofing, pump overrides. This is the standard benchmark for multivariate time series anomaly detection in industrial control systems.

**Why use this.** Scale (51 channels, nearly 1M rows) forces you to confront computational constraints. The attack scenarios create subtle multivariate anomalies where individual sensors may look normal but the cross-sensor relationships are violated — exactly the scenario where ML outperforms thresholds. The labeled attack segments enable rigorous evaluation.

**Download.** Request access at https://itrust.sutd.edu.sg/itrust-labs_datasets/ (free for research, requires registration). The dataset is widely mirrored on Kaggle.

---

## Project structure: seven modules, each building on the last

Each module introduces one technique, applies it to the appropriate dataset, and compares results against everything you've built before. The key discipline: **never move to the next module until you can articulate why the previous technique fails on specific examples.** That failure analysis is where the real learning happens.

---

## Module 1: Fixed thresholds and visual inspection

**Dataset:** NAB `machine_temperature_system_failure.csv`

**What you'll build.** Load the time series, plot it, and manually identify anomalies by eye. Then implement the simplest possible detection: a fixed upper and lower threshold. This is what ISO 10816 severity charts do — alarm if the value exceeds a fixed limit.

**Implementation.**

```python
import pandas as pd
import numpy as np

# Load and plot
df = pd.read_csv('machine_temperature_system_failure.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Fixed threshold: alarm if temperature > X
threshold_high = 90  # picked by looking at the data
df['anomaly_fixed'] = (df['value'] > threshold_high).astype(int)
```

**What to measure.** Precision (what fraction of alarms are real anomalies), recall (what fraction of real anomalies you caught), and F1. Plot the time series with your threshold line and the ground truth anomaly windows. Count false alarms in the normal region.

**What you'll learn.** Three things will become immediately obvious. First, picking the right threshold is hard — too low means constant false alarms, too high means missed detections. Second, a single fixed threshold can't handle drift, seasonality, or changing baselines. Third, point anomalies (sudden spikes) are easy to catch; contextual anomalies (unusual patterns at normal amplitude) are invisible to thresholds.

**Production connection.** This is what Emerson's AMS ships with: pre-programmed limits per equipment type. It catches severe faults reliably but generates alert fatigue on anything subtle.

---

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

---

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

---

## Module 4: Isolation Forest and One-Class SVM — classical ML

**Dataset:** SKAB (all experiments)

**What you'll build.** Two classical ML approaches that learn the boundary of "normal" from unlabeled data.

**Isolation Forest** builds random trees and scores anomalies by path length — anomalies are isolated quickly (short paths) because they're rare and different. **One-Class SVM** maps data to kernel space and finds a tight boundary around the normal data.

**Implementation.**

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler().fit(train[features])
X_train = scaler.transform(train[features])

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,  # expect ~2% anomalies
    random_state=42
)
iso_forest.fit(X_train)

# Score test data
X_test = scaler.transform(test[features])
test['iso_score'] = -iso_forest.score_samples(X_test)  # higher = more anomalous

# One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
ocsvm.fit(X_train)
test['ocsvm_score'] = -ocsvm.decision_function(X_test)
```

**Critical experiment.** Compare Isolation Forest, OCSVM, and Mahalanobis distance across all SKAB anomaly types. You'll likely find that Mahalanobis performs competitively on many anomaly types — confirming the Zope et al. (2019) finding that simple statistical methods match ML in many industrial scenarios. Isolation Forest will likely outperform on the cavitation and rotor imbalance experiments where the anomaly manifests as a nonlinear multivariate shift that Mahalanobis (which assumes Gaussian/linear correlations) misses.

**Hyperparameter sensitivity analysis.** Sweep `contamination` from 0.005 to 0.1 and plot precision/recall curves. Sweep OCSVM's `nu` and `gamma`. Document how sensitive each method is — this is a key production consideration. Isolation Forest is robust; OCSVM is notoriously sensitive.

**What you'll learn.** Isolation Forest's key insight is that anomalies don't need to be far from normal — they just need to be *different* in a way that makes them easy to separate. This is a fundamentally different philosophy from distance-based methods (Mahalanobis, OCSVM). You'll also see that OCSVM's training time grows quadratically with data size — a practical showstopper at production scale.

**Production connection.** Isolation Forest is the most deployed classical ML anomaly detector in production PdM, largely because it runs on edge hardware (ESP32 microcontrollers) and has minimal hyperparameter sensitivity. STMicroelectronics' NanoEdge AI Studio includes Isolation Forest as a default model.

---

## Module 5: Autoencoder — learning to reconstruct "normal"

**Dataset:** SKAB for training and initial testing, then SWaT for scale

**What you'll build.** A standard feedforward autoencoder trained only on healthy data. The reconstruction error becomes your anomaly score.

**Architecture.**

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# Training loop (abbreviated)
model = Autoencoder(input_dim=len(features), latent_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    for batch in train_loader:
        x_hat, _ = model(batch)
        loss = criterion(x_hat, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Anomaly scoring
with torch.no_grad():
    x_hat, z = model(X_test_tensor)
    reconstruction_error = ((X_test_tensor - x_hat) ** 2).mean(dim=1)
```

**Three critical experiments to run:**

*Experiment A — Threshold selection.* Compute reconstruction error on a healthy validation set. Set thresholds at the 95th, 99th, and 99.9th percentile. Run on test data and plot precision-recall curves. This is the exercise that builds your intuition for the false alarm / missed detection tradeoff.

*Experiment B — Latent space visualization.* Extract the latent vectors `z` for both healthy and anomalous data. Project to 2D with t-SNE or UMAP. Healthy points should cluster tightly; anomalous points should scatter away from the cluster. Color by anomaly type (valve, imbalance, cavitation) — do different fault types map to different regions of latent space? If yes, the autoencoder is learning something meaningful about fault modes, not just "different from normal."

*Experiment C — Bottleneck dimension.* Train autoencoders with latent_dim = 2, 4, 8, 16, 32. Plot training loss and anomaly detection F1 for each. Too small → underfits normal data, high reconstruction error on everything. Too large → memorizes everything including anomalies, low reconstruction error on anomalies. The sweet spot teaches you about the information-theoretic capacity of "normal" — how many dimensions does healthy operation actually need?

**What you'll learn.** The autoencoder is learning a compressed model of normal operation. When it encounters an anomaly, it tries to reconstruct it using only the vocabulary of "normal" patterns — and fails, producing high reconstruction error. This is conceptually identical to what a human analyst does: they've internalized what normal looks like, and anomalies jump out because they don't fit the mental model. The bottleneck dimension experiment reveals how complex "normal" really is.

**Production connection.** This is the core architecture of SKF Enlight AI and Siemens Senseye. The threshold selection exercise maps directly to configuring alert severity levels (green/yellow/red) in production dashboards.

---

## Module 6: LSTM-Autoencoder — adding temporal memory

**Dataset:** SKAB and SWaT

**What you'll build.** Replace the feedforward layers with LSTM layers so the model learns temporal sequences, not just individual snapshots. Feed windows of consecutive timesteps and reconstruct the entire window.

**Architecture.**

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16, n_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, n_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        enc_out, (h, c) = self.encoder(x)
        z = self.to_latent(h[-1])              # compress final hidden state
        z_expanded = self.from_latent(z)
        z_repeated = z_expanded.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(z_repeated)
        return dec_out, z
```

**Windowing strategy.** Create sliding windows of length L from the time series. L is critical — experiment with L=10, 30, 60, 120 timesteps. Short windows detect rapid changes (like valve closures); long windows detect slow drift (like gradual rotor imbalance). The reconstruction error is now per-window, and you can compute per-timestep error by averaging across overlapping windows.

**Critical experiment: LSTM-AE vs. static AE on the SKAB "other" experiments.** The `other/6.csv` file contains a *linear* rotor imbalance — a slow, steady increase. The `other/5.csv` contains a *sharp* rotor imbalance — a sudden jump. Run both the static autoencoder (Module 5) and the LSTM-AE on both. The static AE should detect the sharp change easily but struggle with the linear drift. The LSTM-AE should detect the drift earlier because it recognizes the sustained upward trajectory as abnormal, even when individual readings are within normal bounds. Quantify the early warning time difference.

**What you'll learn.** The LSTM learns temporal dependencies — it expects that sensor readings at time t should be consistent with what it saw at t-1, t-2, ..., t-L. A slow drift that stays within the normal range of individual values violates the temporal pattern. This is why LSTM-AEs detect degradation trends that static methods miss, and why they're the "upgrade path" in production systems.

**Production connection.** Advanced cloud PdM platforms use temporal models to catch trending degradation. The sequence length parameter L maps directly to the "lookback window" configuration in production systems — and getting it wrong is a common deployment failure mode.

---

## Module 7: Variational Autoencoder — probabilistic anomaly detection

**Dataset:** SKAB and SWaT

**What you'll build.** Replace the deterministic autoencoder with a VAE that maps inputs to a probability distribution in latent space.

**Architecture.**

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

# VAE loss = reconstruction + KL divergence
def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = nn.MSELoss()(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl
```

**Dual anomaly scoring.** The VAE gives you two anomaly signals: reconstruction error (same as the standard AE) and KL divergence (how far the encoding deviates from the learned normal distribution). Try three scoring strategies: (a) reconstruction error only, (b) KL divergence only, (c) weighted combination. You'll find that different anomaly types trigger different signals — a novel pattern might have high RE but normal KL; a familiar-looking pattern from an unexpected state might have normal RE but high KL.

**β-VAE experiment.** Train with β = 0.1, 0.5, 1.0, 2.0, 5.0. Low β emphasizes reconstruction (the model acts more like a standard AE). High β emphasizes the latent space structure (more disentangled representations but worse reconstruction). Plot both reconstruction quality and anomaly detection performance as functions of β. The optimal β for anomaly detection is typically different from the standard β=1.

**What you'll learn.** The VAE's probabilistic framing gives you uncertainty quantification — not just "is this anomalous?" but "how confident am I that this is anomalous?" This matters for graduated alerting: a high-confidence anomaly triggers an immediate alarm; a low-confidence anomaly triggers a "monitor closely" flag. The β experiment reveals the tension between reconstruction fidelity and latent space structure.

**Production connection.** Uncertainty quantification is what enables the caution → warning → danger alert tiers in production systems. A system that can say "I'm 95% sure this is a fault" drives more trust than one that says "anomaly detected" with no confidence measure.

---

## Module 8: Comparative evaluation and synthesis

**Dataset:** All three, with final focus on SKAB

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
6. Standard Autoencoder
7. LSTM-Autoencoder
8. VAE

**Visualization.** Create a heatmap where rows are methods, columns are SKAB anomaly types (valve1, valve2, rotor imbalance, fluid leak, cavitation), and cell color represents F1 score. This single visualization will crystallize the key insight: simple methods work well on *some* anomaly types (especially sudden process changes), while ML methods earn their keep on *others* (gradual drift, subtle multivariate shifts).

**Health indicator construction.** Take the anomaly scores from your best-performing models and transform them into smooth, monotonic health indicators using the EWMA + isotonic regression pipeline from the previous project guide. Compare the health indicator quality (monotonicity, trendability) across methods.

**Write-up.** Produce a 1-page summary answering: "If I could only deploy one method, which would it be and why?" The answer will depend on constraints — for edge deployment with <1KB RAM, Isolation Forest wins. For cloud deployment on critical assets, LSTM-AE wins. For cold-start with no training data, Mahalanobis with fleet-average baselines wins. This is the PM judgment the project builds.

---

## Implementation order and time estimates

| Module | Time | Key deliverable |
|--------|------|-----------------|
| 1. Fixed thresholds | 2–3 hrs | Baseline F1 on NAB; visceral understanding of false alarm tradeoff |
| 2. Statistical baselines | 3–4 hrs | z-score, EWMA, CUSUM compared; window size sensitivity analysis |
| 3. Mahalanobis distance | 3–4 hrs | First multivariate method on SKAB; comparison with univariate z-scores |
| 4. Isolation Forest + OCSVM | 3–4 hrs | Classical ML vs. Mahalanobis comparison; hyperparameter sensitivity |
| 5. Autoencoder | 4–5 hrs | Reconstruction error anomaly detection; latent space visualization; bottleneck experiment |
| 6. LSTM-Autoencoder | 4–5 hrs | Temporal model vs. static model on slow drift; sequence length analysis |
| 7. VAE | 3–4 hrs | Dual anomaly scoring; β-VAE experiment; uncertainty quantification |
| 8. Comparative evaluation | 4–5 hrs | Full comparison table; health indicator construction; synthesis write-up |

**Total: ~27–34 hours over 3–4 weeks**

---

## What to read alongside each module

**Module 1–2:** Shewhart (1931) on control charts (conceptual); Page (1954) on CUSUM; Roberts (1959) on EWMA. You don't need to read the original papers — the JMP Statistical Knowledge Portal has excellent plain-language explanations at https://www.jmp.com/en/statistics-knowledge-portal/quality-and-reliability-methods/control-charts/cusum-and-ewma-control-charts

**Module 3:** The Hotelling (1947) T² statistic is covered in any multivariate statistics textbook. Focus on understanding why the covariance matrix matters — it's the mathematical encoding of "how do these sensors normally relate to each other?"

**Module 4:** Liu, Ting & Zhou (2008) "Isolation Forest" — https://dl.acm.org/doi/10.1109/ICDM.2008.17. Short, elegant, and the core insight (anomalies are easier to isolate) is immediately useful intuition.

**Module 5:** No single paper — the use of autoencoders for anomaly detection emerged gradually. An & Cho (2015) "Variational Autoencoder Based Anomaly Detection Using Reconstruction Probability" is the closest to a landmark for VAE-based detection.

**Module 6:** Hundman et al. (2018) "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" — KDD 2018. The NASA paper that introduced the SMAP/MSL datasets and demonstrated LSTM-based anomaly detection on real spacecraft telemetry. Practical and well-written. https://arxiv.org/abs/1802.04431

**Module 7:** Kingma & Welling (2014) "Auto-Encoding Variational Bayes" — https://arxiv.org/abs/1312.6114. The foundational VAE paper.

**Overarching survey:** Chandola, Banerjee & Kumar (2009) "Anomaly Detection: A Survey" — https://dl.acm.org/doi/10.1145/1541880.1541882. Read the taxonomy sections (point/contextual/collective anomalies) before starting Module 1 — it gives you the conceptual framework for understanding why different anomaly types require different detection approaches.

**Deep learning-specific survey:** Zamanzadeh Darban et al. (2024) "Deep Learning for Time Series Anomaly Detection: A Survey" — https://dl.acm.org/doi/10.1145/3691338. Read after Module 5 to contextualize your autoencoder work within the broader landscape.

---

## References

### Datasets

1. **NAB (Numenta Anomaly Benchmark)** — https://github.com/numenta/NAB
2. **SKAB (Skoltech Anomaly Benchmark)** — https://github.com/waico/SKAB — also https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab
3. **SWaT (Secure Water Treatment)** — https://itrust.sutd.edu.sg/itrust-labs_datasets/
4. **TS-AD-Datasets (comprehensive list)** — https://github.com/elisejiuqizhang/TS-AD-Datasets
5. **TSB-UAD Benchmark Suite** — https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf
6. **ts-anomaly-benchmark (Zamanzadeh)** — https://github.com/zamanzadeh/ts-anomaly-benchmark

### Foundational surveys

7. Chandola, Banerjee & Kumar (2009). "Anomaly Detection: A Survey." *ACM Computing Surveys*, 41(3). — https://dl.acm.org/doi/10.1145/1541880.1541882
8. Zamanzadeh Darban, Webb, Pan, Aggarwal & Salehi (2024). "Deep Learning for Time Series Anomaly Detection: A Survey." *ACM Computing Surveys*. — https://dl.acm.org/doi/10.1145/3691338
9. Pang, Shen, Cao & van den Hengel (2021). "Deep Learning for Anomaly Detection: A Review." *ACM Computing Surveys*.
10. (2024). "Online model-based anomaly detection in multivariate time series: Taxonomy, survey, research challenges and future directions." *ScienceDirect*. — https://www.sciencedirect.com/science/article/pii/S1574013725000632

### Technique-defining papers

11. Liu, Ting & Zhou (2008). "Isolation Forest." *IEEE ICDM*. — https://dl.acm.org/doi/10.1109/ICDM.2008.17
12. Schölkopf, Platt, Shawe-Taylor, Smola & Williamson (2001). "Estimating the Support of a High-Dimensional Distribution." *Neural Computation*, 13(7). — https://doi.org/10.1162/089976601750264965
13. Kingma & Welling (2014). "Auto-Encoding Variational Bayes." *ICLR*. — https://arxiv.org/abs/1312.6114
14. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." *Neural Computation*, 9(8).
15. An & Cho (2015). "Variational Autoencoder Based Anomaly Detection Using Reconstruction Probability." *Special Lecture on IE*, 2(1).
16. Ruff et al. (2018). "Deep One-Class Classification." *ICML*.
17. Hawkins (1980). *Identification of Outliers*. Chapman & Hall.

### Statistical foundations

18. Page (1954). "Continuous Inspection Schemes." *Biometrika*. (CUSUM)
19. Roberts (1959). "Control Chart Tests Based on Geometric Moving Averages." *Technometrics*. (EWMA)
20. Hotelling (1947). "Multivariate Quality Control." (T² statistic)
21. JMP Statistical Knowledge Portal — CUSUM and EWMA Control Charts. — https://www.jmp.com/en/statistics-knowledge-portal/quality-and-reliability-methods/control-charts/cusum-and-ewma-control-charts

### Applied papers

22. Hundman et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." *KDD 2018*. — https://arxiv.org/abs/1802.04431
23. Rüttinger et al. (2021). "Autoencoder-based Condition Monitoring and Anomaly Detection Method for Rotating Machines." *IEEE*. — https://arxiv.org/pdf/2101.11539
24. Zope et al. (2019). "Anomaly Detection and Diagnosis in Manufacturing Systems." *PHM Conference*. — https://papers.phmsociety.org/index.php/phmconf/article/view/815
25. Katser & Kozitsin (2020). "Skoltech Anomaly Benchmark (SKAB)." *Kaggle*. — https://www.kaggle.com/dsv/1693952

### Benchmark and evaluation papers

26. Paparrizos et al. (2022). "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection." *VLDB*, 15(8). — https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf
27. Wu & Keogh (2021). "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress." *IEEE TKDE*. (Critical evaluation of benchmark datasets)
28. (2024). "TimeSeriesBench: An Industrial-Grade Benchmark for Time Series Anomaly Detection Models." — https://arxiv.org/html/2402.10802v1

### Software and tools

29. Salesforce Merlion — Time series anomaly detection library with unified dataset loading. — https://opensource.salesforce.com/Merlion/
30. scikit-learn — Novelty and Outlier Detection documentation. — https://scikit-learn.org/stable/modules/outlier_detection.html
31. PyOD — Python Outlier Detection library (30+ algorithms). — https://github.com/yzhao062/pyod

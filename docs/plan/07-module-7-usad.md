## Module 7: USAD — UnSupervised Anomaly Detection on Multivariate Time Series

**Dataset:** SWaT (primary), NASA MSL (secondary)

**Status: Not started**

**Reference paper:** Audibert et al. (2020). "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." *KDD 2020*.

**Reference implementation:** [https://github.com/liuziyi/USAD](https://github.com/liuziyi/USAD)

---

### What USAD is

USAD combines an autoencoder architecture with adversarial training to produce anomaly scores that are both sensitive and stable. The key architectural insight: two decoders share one encoder, and they are trained in two competing phases.

**Architecture:**

```
Input W → Encoder E → latent Z
                    → Decoder D1 → W1
                    → Decoder D2 → W2
```

**Training (single loop, Algorithm 1):**

Both losses are computed in the same forward pass each iteration. `n` is the epoch counter (1 to N) — it appears in the loss coefficients so the adversarial term weight `(1 − 1/n)` grows from 0 toward 1 as training progresses. Two separate optimizers: one for AE1 (E+D1), one for AE2 (D2).

For each batch window W, each epoch n:
- `Z = E(W)`
- `W1' = D1(Z)` — AE1 reconstruction
- `W2' = D2(Z)` — AE2 reconstruction
- `W2'' = D2(E(W1'))` — AE2 applied to AE1's output

Loss for AE1 (E+D1): `ℒ_AE1 = (1/n)||W − W1'||₂ + (1 − 1/n)||W − W2''||₂`
Loss for AE2 (D2):   `ℒ_AE2 = (1/n)||W − W2'||₂ − (1 − 1/n)||W − W2''||₂`

AE1 minimizes both its own reconstruction error *and* the adversarial term (making W1' hard for AE2 to distinguish from real data). AE2 minimizes its own reconstruction error but *maximizes* the adversarial term (learning to detect AE1's outputs). The `(1 − 1/n)` schedule starts at 0 (pure reconstruction) and ramps to 1 (full adversarial weight) over training.

**Anomaly score at inference:**

```
score(W) = α · ||W − D1(E(W))||₂ + β · ||W − D2(E(D1(E(W))))||₂
```

The α/β trade-off controls sensitivity vs. specificity: higher β amplifies anomaly signal but increases false alarms on normal data. Sweep α+β=1 to find the operating point.

---

### Why USAD after LSTM-AE

The project has established:
- Feedforward AE (Module 5): F1 0.7264 on SWaT
- LSTM-AE (Module 6): F1 0.7533 on SWaT — best result so far
- LSTM-AE on MSL (Module 6 extension): F1 0.1292 micro — negative result, underperformed FF-AE

USAD is the natural next step for two reasons:

1. **Adversarial amplification.** The two-decoder adversarial phase directly addresses the core weakness of standard AEs: normal data and mildly anomalous data produce similar reconstruction errors, making threshold calibration brittle. USAD's second loss term explicitly amplifies the gap.

2. **Window-based, not sequential.** USAD uses sliding windows (no recurrence), making it tractable on MSL's sparse per-channel training sets where LSTM-AE failed. This tests whether the adversarial training mechanism can recover what temporal modeling could not.

---

### What you'll build

#### 1. Implement USAD from scratch (SWaT)

Implement the two-phase training loop in PyTorch following the paper. Key implementation details:

- **Data source.** Load `merged.csv` only — it contains the full 11-day dataset with `Normal/Attack` labels. The raw file has ~495K duplicate rows (same timestamp, same values); deduplicate by timestamp (drop duplicates, sort, reset index) to get 946,719 unique rows.
- **Train/test split.** First 496,800 rows = 7-day normal period (training); remaining 449,919 rows = 4-day attack period (test). This matches the paper's split exactly. Training data has 0% attack rate; test data has ~12% attack rate (paper reports 11.98%).
- **Downsampling.** Apply block-median downsampling (factor 5) to **both** train and test: for each consecutive block of 5 rows, take the median value per feature. Train: 496,800 → 99,360 rows. Test: 449,919 → ~89,983 rows. Downsampling both sets is required for consistency — the model trains on 5-second block medians (W=12 covers 60 seconds of real time); presenting raw 1-second samples at inference would create a 5× timescale mismatch and a different feature distribution. **Label rule for test blocks:** OR (max) — a block is labeled "Attack" if any of its 5 constituent rows is an attack, so short anomalies are not swallowed by aggregation. Do not use simple decimation (`[::5]`) — use reshape + median.
- **Window construction.** Slide a window of length `w=12` over the downsampled, normalized data. Each sample is a flattened window vector of size `w × n_features = 12 × 51 = 612`.
- **Architecture.** Three linear layers per encoder/decoder (Appendix A.4). Encoder: `[612 → 306 → 153 → 40]` with ReLU after each layer. Decoders D1, D2: `[40 → 153 → 306 → 612]` with ReLU after the first two layers and Sigmoid on the output. Use `z_dim=40` (paper default). Layer sizes follow the paper's halving rule: `input_size=612`, `input_size/2=306`, `input_size/4=153`.
- **Training loop.** Single loop — both losses computed per batch, two separate Adam optimizers (one for E+D1, one for E+D2), both stepped each iteration. E is shared between both AEs and must appear in both optimizers. Train for 70 epochs (paper default for SWaT). Use Adam with default lr (paper specifies "default learning rate").
- **Loss for AE1 (optimizer 1 — E+D1):** `ℒ_AE1 = (1/n)||W − D1(E(W))||₂ + (1 − 1/n)||W − D2(E(D1(E(W))))||₂`
- **Loss for AE2 (optimizer 2 — E+D2):** `ℒ_AE2 = (1/n)||W − D2(E(W))||₂ − (1 − 1/n)||W − D2(E(D1(E(W))))||₂`
- where `n` is the current epoch (1-indexed), so the adversarial weight `(1 − 1/n)` ramps from 0 to ~1 over training.
- **Anomaly score.** For each window, compute `score = α·||W−D1(E(W))||₂ + β·||W−D2(E(D1(E(W))))||₂` with `α=0.1, β=0.9` as starting point.

#### 2. Threshold sweep (α/β and percentile)

Two threshold axes to sweep:

- **α/β ratio:** Try `(α,β) ∈ {(1,0), (0.5,0.5), (0.1,0.9), (0.01,0.99)}`. Plot F1 vs. β to understand the adversarial amplification effect.
- **Score percentile:** For each α/β, sweep percentile threshold from p95 to p99.9. Report best F1 per configuration.

Expected finding: higher β should improve F1 on SWaT (attack windows are genuinely anomalous and the adversarial term should amplify them), but may hurt on MSL (sparse anomalies mixed with gradual drift).

#### 3. Compare against project baselines on SWaT

Reproduce the full comparison table for SWaT:

| Method              | F1     | Notes                            |
| ------------------- | ------ | -------------------------------- |
| Mahalanobis         | 0.7240 | Module 3 result                  |
| Isolation Forest    | 0.5154 | Module 4 result                  |
| Feedforward AE      | 0.7264 | Module 5 result (p99.9)          |
| LSTM-AE             | 0.7533 | Module 6 result (p99.9, 40 ep)   |
| USAD                | TBD    | This module                      |

Key question: does the adversarial training mechanism push USAD above LSTM-AE's 0.7533, or does the absence of temporal modeling limit it on SWaT's slow drift anomalies?

#### 4. Apply USAD to NASA MSL

Run USAD in two modes on MSL:

- **Per-channel mode:** Train one USAD model per MSL channel (27 models), matching the LSTM-AE evaluation from Module 6. Use `w=30` windows (more context than the 12-step default; MSL channels have longer sequences).
- **Multi-channel mode:** Train a single USAD model on all 27 MSL channels concatenated, treating MSL as a multivariate dataset. Compare micro-F1 against per-channel LSTM-AE (0.1292).

This directly tests the hypothesis that USAD's adversarial amplification can compensate for the sparse training windows that defeated LSTM-AE on MSL.

---

### Key diagnostic plots

Produce the following visualizations:

1. **Training curves.** Phase 1 and phase 2 losses over epochs for SWaT. Confirm phase 2 divergence pattern (D1 and D2 losses should pull against each other after phase 1 converges).

2. **Score distribution.** Histogram of anomaly scores for normal vs. attack windows on SWaT. Compare separation to LSTM-AE score distribution from Module 6. Quantify as `(μ_attack − μ_normal) / σ_normal`.

3. **α/β sensitivity.** Line plot of F1 vs. β (with α=1−β) on SWaT. Expected shape: F1 rises with β up to a peak, then drops as false alarms accumulate.

4. **Time series overlay.** Plot USAD anomaly scores against SWaT attack labels for 3 representative attack episodes. Compare visually to LSTM-AE scores from Module 6.

5. **MSL per-channel F1 bar chart.** Bar chart of per-channel F1 for USAD vs. LSTM-AE across all 27 MSL channels, sorted by channel difficulty.

---

### Expected outcomes and learning objectives

**If USAD beats LSTM-AE on SWaT:** The adversarial amplification mechanism adds real value beyond temporal modeling. The key lesson is that the *loss function design* (adversarial term) matters as much as the *architecture choice* (LSTM vs. MLP).

**If USAD matches LSTM-AE on SWaT:** The adversarial training and temporal modeling are complementary — neither dominates. SWaT's slow drift anomalies may inherently require temporal context that windowed USAD can't capture.

**If USAD beats LSTM-AE on MSL:** Validates that adversarial amplification is the missing ingredient for sparse-label settings. This would be a stronger positive result than SWaT, since MSL is the harder benchmark.

**If USAD underperforms on both:** Suggests the adversarial mechanism requires more training data than the project datasets provide, or that the window length `w` is a sensitive hyperparameter worth ablating.

---

### Deliverables

- `notebooks/07-swat-usad.ipynb` — USAD on SWaT: implementation, training, threshold sweep, comparison table
- `notebooks/07-msl-usad.ipynb` — USAD on MSL: per-channel and multi-channel modes, comparison to LSTM-AE per-channel results
- Updated status table in `00-project-overview.md`

---

### Reading

- Audibert, Michiardi, Guyard, Marti & Zuluaga (2020). "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." *KDD 2020*. — the primary reference; read Section 3 (architecture) and Section 4 (experiments on SWaT and SMAP/MSL) before implementing.
- Reference implementation: [https://github.com/liuziyi/USAD](https://github.com/liuziyi/USAD) — use for architecture validation and hyperparameter defaults, but implement from scratch to build understanding.

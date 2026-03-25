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

**Two-phase training:**

- **Phase 1 (AE reconstruction):** Train E+D1 and E+D2 independently as standard autoencoders. Loss = ||W − D1(E(W))||² + ||W − D2(E(W))||²
- **Phase 2 (adversarial):** Train E+D1 to fool D2 (minimize D2's ability to detect reconstruction errors). Train D2 to detect when its input came from D1 rather than the real encoder. Loss introduces a term that amplifies anomaly scores while suppressing false alarms on normal data.

**Anomaly score at inference:**

```
score(W) = α · ||W − D1(E(W))||² + β · ||W − D2(D1(E(W)))||²
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

- **Window construction.** Slide a window of length `w` (start with `w=12`, matching the paper) over the normalized training data. Each sample is a flattened window vector of size `51 × w`.
- **Architecture.** Encoder: `[51×w → 512 → 256 → z_dim]`. Decoders D1, D2: `[z_dim → 256 → 512 → 51×w]`. Use `z_dim=40` (paper default). ReLU activations throughout.
- **Phase 1 loss:** `L1 = (1/n) Σ ||W − D1(E(W))||² + ||W − D2(E(W))||²`
- **Phase 2 loss:** `L2 = (1/n) Σ ||W − D1(E(W))||² − ||W − D2(D1(E(W)))||²` for E+D1; `L3 = (1/n) Σ ||W − D2(D1(E(W)))||²` for D2
- **Training schedule.** Alternate phase 1 and phase 2 each epoch. Train for 100 epochs (paper default). Use Adam with lr=1e-4.
- **Anomaly score.** For each window, compute `score = α·||W−D1(E(W))||² + β·||W−D2(D1(E(W)))||²` with `α=0.1, β=0.9` as starting point.

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

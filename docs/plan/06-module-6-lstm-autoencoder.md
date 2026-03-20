## Module 7: LSTM-Autoencoder — adding temporal memory

**Dataset:** SKAB for initial testing, NASA SMAP/MSL as primary benchmark

**Status: In progress** — SWaT notebook at `notebooks/06-swat-lstm-ae.ipynb`

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

**Critical experiment: LSTM-AE vs. static AE on the SKAB "other" experiments.** The `other/6.csv` file contains a *linear* rotor imbalance — a slow, steady increase. The `other/5.csv` contains a *sharp* rotor imbalance — a sudden jump. Run both the static autoencoder (Module 6) and the LSTM-AE on both. The static AE should detect the sharp change easily but struggle with the linear drift. The LSTM-AE should detect the drift earlier because it recognizes the sustained upward trajectory as abnormal, even when individual readings are within normal bounds. Quantify the early warning time difference.

**Primary benchmark on NASA SMAP/MSL.** The NASA SMAP/MSL dataset is the ideal showcase for LSTM temporal modeling:
- **SMAP (Soil Moisture Active Passive):** 25-channel spacecraft telemetry, ~500K timesteps, 55 labeled anomaly sequences. Contains both point anomalies and slow contextual drift.
- **MSL (Mars Science Laboratory):** 55-channel rover telemetry, similar scale. The multi-channel nature requires the model to learn cross-sensor temporal patterns.
- These datasets contain exactly the gradual drift and contextual anomalies that static methods miss but LSTM-AEs excel at.
- Compare against Module 6 static autoencoder on the same NASA channels. The delta F1 quantifies the value of temporal modeling on real spacecraft data.

The original paper (Hundman et al. 2018) is the benchmark to compare against — they report per-entity F1 scores for each SMAP/MSL channel, enabling direct comparison.

**What you'll learn.** The LSTM learns temporal dependencies — it expects that sensor readings at time t should be consistent with what it saw at t-1, t-2, ..., t-L. A slow drift that stays within the normal range of individual values violates the temporal pattern. This is why LSTM-AEs detect degradation trends that static methods miss, and why they're the "upgrade path" in production systems.

**Production connection.** Advanced cloud PdM platforms use temporal models to catch trending degradation. The sequence length parameter L maps directly to the "lookback window" configuration in production systems — and getting it wrong is a common deployment failure mode.

**Reading.** Hundman et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." *KDD 2018.* — [https://arxiv.org/abs/1802.04431](https://arxiv.org/abs/1802.04431). This is the paper that introduced the SMAP/MSL datasets and established the LSTM-AE baseline you're reproducing.

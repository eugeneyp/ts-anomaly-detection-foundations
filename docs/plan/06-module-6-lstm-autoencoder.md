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

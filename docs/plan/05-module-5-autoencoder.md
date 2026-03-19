## Module 6: Autoencoder — learning to reconstruct "normal"

**Dataset:** SKAB for initial testing, NASA SMAP/MSL for scale benchmark

**Status: Not started**

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

**Scale benchmark on NASA SMAP/MSL.** After validating the architecture on SKAB (8 channels), scale to the NASA SMAP/MSL dataset (25–55 channels, 500K+ timesteps per entity). NASA provides a realistic test of whether reconstruction-based detection generalizes to longer sequences and more channels. Compare autoencoder F1 on the NASA benchmark against prior methods (Mahalanobis, IF, RCF) where applicable.

**What you'll learn.** The autoencoder is learning a compressed model of normal operation. When it encounters an anomaly, it tries to reconstruct it using only the vocabulary of "normal" patterns — and fails, producing high reconstruction error. This is conceptually identical to what a human analyst does: they've internalized what normal looks like, and anomalies jump out because they don't fit the mental model. The bottleneck dimension experiment reveals how complex "normal" really is.

**Production connection.** This is the core architecture of SKF Enlight AI and Siemens Senseye. The threshold selection exercise maps directly to configuring alert severity levels (green/yellow/red) in production dashboards.

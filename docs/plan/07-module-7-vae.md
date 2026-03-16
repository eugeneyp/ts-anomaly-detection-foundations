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

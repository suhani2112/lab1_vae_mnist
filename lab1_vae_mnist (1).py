"""Lab 1 - Variational Autoencoder for Image Generation (MNIST).

ETMMGA304 Laboratory Manual, pages 4-6.

Trains a VAE with a two-dimensional latent space on MNIST, logs the
per-image reconstruction (BCE) and KL-divergence terms, and saves:

    lab_outputs/vae_mnist.pt
    lab_outputs/vae_reconstructed.png
    lab_outputs/vae_generated.png
    lab_outputs/vae_latent.png

Run:
    python lab1_vae_mnist.py [--epochs 5] [--batch-size 128]
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend: works in terminals, CI and Colab.

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

SEED = 42
OUT_DIR = Path("lab_outputs")
DATA_DIR = Path("data")
LATENT_DIM = 2


# --------------------------------------------------------------------------- #
# Common setup
# --------------------------------------------------------------------------- #
def setup(seed: int = SEED) -> torch.device:
    """Seed RNGs, create output folders and return the compute device."""
    random.seed(seed)
    torch.manual_seed(seed)
    OUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")
    return device


# --------------------------------------------------------------------------- #
# Step 1 - Data
# --------------------------------------------------------------------------- #
def load_mnist() -> tuple[Dataset, Dataset]:
    """Download (if needed) and return the MNIST train and test splits."""
    to_tensor = transforms.ToTensor()
    train = datasets.MNIST(str(DATA_DIR), train=True, download=True,
                           transform=to_tensor)
    test = datasets.MNIST(str(DATA_DIR), train=False, download=True,
                          transform=to_tensor)
    return train, test


# --------------------------------------------------------------------------- #
# Step 2 - Model
# --------------------------------------------------------------------------- #
class VAE(nn.Module):
    """Fully connected VAE: 784 -> 256 -> (mu, logvar) in R^2 -> 256 -> 784."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                                     nn.ReLU())
        self.mean = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(),
                                     nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mean(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + exp(0.5 * logvar) * eps, with eps ~ N(0, I)."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss_terms(
    generated: torch.Tensor, x: torch.Tensor,
    mu: torch.Tensor, logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed reconstruction BCE and KL divergence for a batch."""
    recon = F.binary_cross_entropy(generated, x.flatten(1), reduction="sum")
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum()
    return recon, kl


# --------------------------------------------------------------------------- #
# Step 3 - Training
# --------------------------------------------------------------------------- #
def train(vae: VAE, loader: DataLoader, device: torch.device,
          epochs: int = 5, lr: float = 0.001) -> list[dict[str, float]]:
    """Train the VAE and return per-image loss terms for every epoch."""
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n_images = len(loader.dataset)  # type: ignore[arg-type]
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        vae.train()
        totals = {"reconstruction": 0.0, "kl": 0.0}
        for x, _ in loader:
            x = x.to(device)
            generated, mu, logvar = vae(x)
            recon, kl = vae_loss_terms(generated, x, mu, logvar)
            loss = (recon + kl) / x.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["reconstruction"] += recon.item()
            totals["kl"] += kl.item()

        per_image = {k: round(v / n_images, 2) for k, v in totals.items()}
        if not all(math.isfinite(v) for v in per_image.values()):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {per_image}")
        history.append({"epoch": epoch, **per_image})
        print(f"Epoch {epoch}: {per_image}")

    torch.save(vae.state_dict(), OUT_DIR / "vae_mnist.pt")
    print(f"Saved checkpoint -> {OUT_DIR / 'vae_mnist.pt'}")
    return history


# --------------------------------------------------------------------------- #
# Step 4 - Sampling and evaluation
# --------------------------------------------------------------------------- #
def show_grid(images: torch.Tensor, name: str, nrow: int = 4) -> None:
    """Save a grid of 1x28x28 images to lab_outputs/<name>."""
    grid = make_grid(images.cpu(), nrow=nrow)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / name}")


@torch.no_grad()
def evaluate(vae: VAE, test: Dataset, device: torch.device) -> None:
    """Save reconstructions, prior samples and the held-out latent scatter."""
    vae.eval()
    x, y = next(iter(DataLoader(test, batch_size=1024)))
    mu, _ = vae.encode(x.to(device))

    recon = vae.decoder(mu[:8]).reshape(-1, 1, 28, 28)
    samples = vae.decoder(torch.randn(16, LATENT_DIM, device=device))
    show_grid(torch.cat([x[:8], recon.cpu()]), "vae_reconstructed.png", nrow=8)
    show_grid(samples.reshape(-1, 1, 28, 28), "vae_generated.png")

    coords = mu.cpu()
    plt.figure(figsize=(6, 4))
    plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=9)
    plt.colorbar(ticks=range(10), label="Digit label")
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("Held-out MNIST latent means")
    plt.savefig(OUT_DIR / "vae_latent.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'vae_latent.png'}")


def print_observation_table(history: list[dict[str, float]]) -> None:
    """Print the rows the lab record asks for (epochs 1, 3 and 5)."""
    print("\nEpoch | Reconstruction / image | KL / image")
    for row in history:
        if int(row["epoch"]) in (1, 3, 5):
            print(f"{int(row['epoch']):>5} | {row['reconstruction']:>22} | "
                  f"{row['kl']:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = setup()
    train_set, test_set = load_mnist()
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    vae = VAE().to(device)
    history = train(vae, loader, device, epochs=args.epochs)
    evaluate(vae, test_set, device)
    print_observation_table(history)


if __name__ == "__main__":
    main()

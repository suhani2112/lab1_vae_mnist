# Assignment 1: Variational Autoencoder for Image Generation

**ETMMGA304 – Generative AI and Its Applications** · MCA, K.R. Mangalam University
· Lab 1 (lab manual pages 4–6)

Builds and trains a VAE with a 2-D latent space on MNIST, compares reconstructions with the original images, samples new digits from the prior, and plots the latent space of held-out images.

**Settings:** Encoder 784→256→(μ, log σ²) ∈ ℝ², decoder 2→256→784 + Sigmoid; Adam lr 0.001, batch 128, 5 epochs, seed 42.

## Files

```
GenAI-Assignment-1/
├── README.md
├── LAB_RECORD.md        # observation tables, results, viva answers
├── requirements.txt
├── lab1_vae_mnist.py
├── lab1_vae_mnist.ipynb
├── data/                # created automatically (datasets)
└── lab_outputs/         # created automatically (results)
```

## Setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/Aman13082001/GenAI-Assignment-1.git
cd GenAI-Assignment-1
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Hardware:** Runs on CPU in about 2–5 minutes. A GPU is faster but not needed.

### CPU or CUDA GPU

On Windows, `pip install -r requirements.txt` installs the CPU build of PyTorch.
For an NVIDIA GPU, install the CUDA build first (choose the CUDA version for
your driver at [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check with `python -c "import torch; print(torch.cuda.is_available())"`.

## Run

```bash
python lab1_vae_mnist.py            # options: --epochs 5 --batch-size 128
```

Outputs:

- `lab_outputs/vae_mnist.pt` (checkpoint)
- `lab_outputs/vae_reconstructed.png`
- `lab_outputs/vae_generated.png`
- `lab_outputs/vae_latent.png`

## Google Colab

Open [lab1_vae_mnist.ipynb](lab1_vae_mnist.ipynb) in Colab (File → Open notebook → GitHub, then paste this repo's URL). Choose **Runtime → Change runtime type → T4 GPU** if you need a GPU, then run the cells in order. Download `lab_outputs/` before the session ends.

## Lab record

The observation tables, expected ranges and viva answers are in
[LAB_RECORD.md](LAB_RECORD.md). Fill the tables in from your own run.

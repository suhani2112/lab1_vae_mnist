# Lab Record: Assignment 1, ETMMGA304 Generative AI and Its Applications

**Student:** ____________________  **Roll no.:** ____________  **Programme:** MCA

Fill in each blank table from **your own run**. The *expected ranges* below
are approximate values for the default settings, to help you spot a broken
run. They are not results to copy. Values change with hardware, library
versions and random seeds.

Write down the following for every experiment:

| Field | Value |
|-------|-------|
| Date | |
| Device (CPU / GPU model) | |
| Python / PyTorch / Transformers versions | |
| Seed | 42 |

---

## Lab 1: Variational Autoencoder for Image Generation

**Aim.** Build and train a VAE in PyTorch on MNIST. Generate handwritten
digits, compare reconstructions with the original images, and visualise the
2-D latent space.

**Software / data.** PyTorch, torchvision MNIST (60k train / 10k test),
Matplotlib. Script: `lab1_vae_mnist.py`.

**Parameters.** Encoder 784→256→(μ, log σ²) ∈ ℝ²; decoder 2→256→784 with a
Sigmoid output. Adam, lr = 0.001, batch size 128, 5 epochs, seed 42.

### Observation table

| Epoch | Reconstruction loss / image | KL divergence / image | Comments |
|-------|-----------------------------|-----------------------|----------|
| 1 | | | |
| 3 | | | |
| 5 | | | |

**Expected ranges.** Reconstruction BCE is about 175–200 per image at epoch 1
and falls to about 145–160 by epoch 5. KL is about 4–6 per image and usually
*rises* slightly over training (to about 6–7), because the encoder spreads
the codes out to carry more information. Both values must stay finite.

| Visual check | Your value |
|--------------|------------|
| Recognisable digits out of 16 (`vae_generated.png`) | / 16 |
| Two labels whose latent points overlap (`vae_latent.png`) | |
| One region with a clear grouping | |

Typical outcome: 8–13 of 16 samples are readable. 4/9 and 3/5/8 overlap in
the latent plot, while 0 and 1 form well-separated clusters.

### Result

Reconstructions keep the digit identity and overall shape but are blurrier
than the originals. Samples drawn from the prior give plausible digits
wherever the prior overlaps the learned clusters.

**Limitation.** A 2-D latent space is too small to hold fine stroke detail,
and samples from regions between clusters look like blends of two digits.
**Next step.** Increase the latent size to 16–32 (and plot it with t-SNE or
PCA), or use a convolutional encoder and decoder.

### Viva answers

1. **What do the mean and log variance represent?**
   They are the parameters of the approximate posterior q(z|x) = 𝒩(μ, σ²).
   μ is the centre of the region of latent space the encoder assigns to the
   image. log σ² is how uncertain that position is. The encoder predicts the
   log variance instead of σ² because log σ² can be any real number, while
   σ² must be positive. This keeps training numerically stable.

2. **Why is the KL term included in the objective?**
   KL(q(z|x) ‖ 𝒩(0, I)) pulls every encoding towards the standard-normal
   prior. Without it the encoder could give each image a tiny, isolated
   code, and the model would become a plain autoencoder. Decoding random
   z ~ 𝒩(0, I) would then land in empty space and give garbage. The KL term
   makes the latent space continuous and densely filled, which is what makes
   sampling work. Together with the reconstruction term, it forms the
   negative ELBO.

3. **How does a reconstructed digit differ from a newly generated digit?**
   A reconstruction decodes μ(x) for a real test image, so it is tied to
   that input and should look like it. A generated digit decodes a fresh
   z ~ 𝒩(0, I) with no input image, so it shows what the model has learned
   about the whole data distribution. Generated digits are usually less
   sharp and sometimes ambiguous.

4. **Why use held-out images to draw the latent scatterplot?**
   Test images show how the encoder handles data it never trained on, so
   the plot shows whether the latent structure generalises. Training images
   could show clusters that only reflect memorisation.

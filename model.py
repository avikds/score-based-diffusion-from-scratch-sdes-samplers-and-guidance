"""
Score-Based Diffusion from Scratch: SDEs, Samplers and Guidance

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - sample_mixture
import torch

def sample_mixture(mix, n):
    # Extract the mixture parameters.
    means = mix["means"]      # Shape: (K, 2)
    stds = mix["stds"]        # Shape: (K,)
    weights = mix["weights"]  # Shape: (K,)

    # Sample component indices according to the mixture weights.
    # Sampling is performed with replacement.
    y = torch.multinomial(weights, n, replacement=True)

    # Draw standard Gaussian noise for each sample.
    z = torch.randn(n, 2, dtype=means.dtype, device=means.device)

    # Generate samples using the selected component's mean and standard deviation.
    # Unsqueeze stds to shape (n, 1) for correct broadcasting over both dimensions.
    x = means[y] + stds[y].unsqueeze(-1) * z

    return x, y

# Step 2 - mixture_responsibilities
import math

def mixture_responsibilities(x, mix):
    means = mix["means"]      # Shape: (K, 2)
    stds = mix["stds"]        # Shape: (K,)
    weights = mix["weights"]  # Shape: (K,)

    # Compute the difference between every point and every component mean.
    # Shape: (n, K, 2)
    diff = x[:, None, :] - means[None, :, :]

    # Squared Euclidean distance from each point to each component mean.
    # Shape: (n, K)
    squared_dist = diff.square().sum(dim=-1)

    # Compute log mixture weights plus isotropic 2-D Gaussian log densities.
    # For a 2-D Gaussian, the log-density normalization is:
    # -log(2*pi) - 2*log(std).
    log_probs = (
        torch.log(weights)[None, :]
        - math.log(2.0 * math.pi)
        - 2.0 * torch.log(stds)[None, :]
        - squared_dist / (2.0 * stds.square()[None, :])
    )

    # Normalize the component log probabilities for each point.
    # Output shape: (n, K); every row sums to one.
    responsibilities = torch.softmax(log_probs, dim=-1)

    return responsibilities

# Step 3 - mixture_score
def mixture_score(x, mix):
    means = mix["means"]  # Shape: (K, 2)
    stds = mix["stds"]    # Shape: (K,)

    # Compute the posterior probability of each component for every point.
    # Shape: (n, K)
    responsibilities = mixture_responsibilities(x, mix)

    # Compute each component's score: (mu_k - x) / std_k^2.
    # Shape: (n, K, 2)
    component_scores = (
        means.unsqueeze(0) - x.unsqueeze(1)
    ) / stds.square().view(1, -1, 1)

    # Combine component scores using their posterior responsibilities.
    # Shape: (n, 2)
    score = (
        responsibilities.unsqueeze(-1) * component_scores
    ).sum(dim=1)

    return score

# Step 4 - perturbed_mixture
def perturbed_mixture(mix, sigma, alpha=1.0):
    means = mix["means"]
    stds = mix["stds"]
    weights = mix["weights"]

    # Scaling the Gaussian samples by alpha scales their means and standard deviations.
    new_means = alpha * means

    # Adding independent Gaussian noise increases the variance by sigma^2.
    # Var(alpha * X + sigma * Z) = alpha^2 * stds^2 + sigma^2.
    new_stds = torch.sqrt((alpha * stds).square() + sigma**2)

    # Return a new dictionary without modifying the original mixture.
    # The mixture weights remain unchanged.
    return {
        "means": new_means,
        "stds": new_stds,
        "weights": weights,
    }

# Step 5 - dsm_loss
def dsm_loss(model, x0, sigma, eps, alpha=1.0):
    # Convert sigma to a tensor on the same device and with the same dtype as x0.
    sigma = torch.as_tensor(sigma, dtype=x0.dtype, device=x0.device)

    # Support either a scalar noise level or one noise level per sample.
    if sigma.ndim == 0:
        sigma = sigma.expand(x0.shape[0])
    else:
        sigma = sigma.reshape(-1)

    sigma_column = sigma.unsqueeze(-1)

    # Support scalar alpha or a separate alpha value for each sample.
    alpha = torch.as_tensor(alpha, dtype=x0.dtype, device=x0.device)
    if alpha.ndim > 0:
        alpha = alpha.reshape(-1, 1)

    # Construct perturbed samples with per-row scaling.
    x_t = alpha * x0 + sigma_column * eps

    # Predict scores conditioned on each sample's noise level.
    score = model(x_t, sigma)

    # Weighted denoising score matching objective.
    residual = sigma_column * score + eps
    loss = 0.5 * residual.square().sum(dim=-1).mean()

    return loss

# Step 6 - ScoreNet
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()

        # Noise-conditioned MLP: input (x1, x2, log(sigma)) -> predicted noise.
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, sigma):
        # Ensure sigma matches the input tensor's dtype and device.
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)

        # Support a scalar noise level or one noise level per sample.
        if sigma.ndim == 0:
            sigma = sigma.expand(x.shape[0])
        else:
            sigma = sigma.reshape(-1)

        # Encode the noise level and concatenate it with the 2-D input.
        log_sigma = torch.log(sigma).unsqueeze(-1)
        features = torch.cat([x, log_sigma], dim=1)

        # Predict the noise-related quantity and convert it to a score.
        # The score is the network output divided by sigma row-wise.
        output = self.net(features)
        score = output / sigma.unsqueeze(-1)

        return score

# Step 7 - train_score_net
def train_score_net(model, mix, noise_fn, n_steps, batch_size, lr):
    # Initialize the optimizer and learning-rate scheduler once.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps
    )

    losses = []
    model.train()

    for _ in range(n_steps):
        # Sample clean data points from the Gaussian mixture.
        x0, _ = sample_mixture(mix, batch_size)

        # Draw per-sample noise schedules: alpha and sigma, each of shape (n,).
        alpha, sigma = noise_fn(batch_size)

        # Sample standard Gaussian noise with the same shape as x0.
        eps = torch.randn_like(x0)

        # Compute the denoising score matching loss.
        loss = dsm_loss(model, x0, sigma, eps, alpha)

        # Perform one optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate after the optimizer step.
        scheduler.step()

        # Store each loss as a Python float.
        losses.append(float(loss.item()))

    return losses


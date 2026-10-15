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

# Step 8 - geometric_sigmas
def geometric_sigmas(sigma_max, sigma_min, n_levels):
    # Handle the single-level case explicitly.
    if n_levels == 1:
        return torch.tensor([sigma_max], dtype=torch.float32)

    # Generate noise levels equally spaced in logarithmic space,
    # decreasing from sigma_max to sigma_min.
    log_sigmas = torch.linspace(
        math.log(sigma_max),
        math.log(sigma_min),
        steps=n_levels
    )

    return torch.exp(log_sigmas)

# Step 9 - langevin_step
def langevin_step(x, score, step_size, z):
    # Perform one unadjusted Langevin dynamics step.
    # The score contributes a drift term, while z adds Gaussian noise.
    x_new = (
        x
        + 0.5 * step_size * score
        + math.sqrt(step_size) * z
    )

    return x_new

# Step 10 - annealed_langevin
def annealed_langevin(score_fn, x_init, sigmas, n_steps_each, eps):
    # Clone the initial points so that x_init remains unchanged.
    x = x_init.clone()

    # The smallest noise level determines the scaling of the step sizes.
    sigma_min = float(sigmas[-1])

    # Process noise levels in decreasing order.
    for sigma_i in sigmas:
        sigma_value = float(sigma_i)

        # Scale the step size according to Song and Ermon's Algorithm 1.
        step_size = eps * (sigma_value / sigma_min) ** 2

        # Create a per-sample noise-level tensor.
        sigma_batch = torch.full(
            (x.shape[0],),
            sigma_value,
            dtype=x.dtype,
            device=x.device
        )

        # Perform the prescribed number of Langevin updates at this level.
        for _ in range(n_steps_each):
            score = score_fn(x, sigma_batch)
            z = torch.randn_like(x)
            x = langevin_step(x, score, step_size, z)

    return x

# Step 11 - gaussian_frechet_distance
def gaussian_frechet_distance(mu1, cov1, mu2, cov2):
    # Compute the squared Euclidean distance between the means.
    mean_term = (mu1 - mu2).square().sum()

    # Symmetrize covariance matrices to reduce numerical asymmetry.
    cov1 = 0.5 * (cov1 + cov1.T)
    cov2 = 0.5 * (cov2 + cov2.T)

    # Compute the symmetric square root of cov1 using eigendecomposition.
    eigvals1, eigvecs1 = torch.linalg.eigh(cov1)
    eigvals1 = eigvals1.clamp_min(0.0)
    sqrt_cov1 = (
        eigvecs1 * torch.sqrt(eigvals1).unsqueeze(0)
    ) @ eigvecs1.T

    # Form the symmetric positive-semidefinite matrix
    # cov1^(1/2) @ cov2 @ cov1^(1/2).
    middle = sqrt_cov1 @ cov2 @ sqrt_cov1
    middle = 0.5 * (middle + middle.T)

    # Its eigenvalues give the trace of the matrix square root.
    eigvals_middle = torch.linalg.eigvalsh(middle)
    eigvals_middle = eigvals_middle.clamp_min(0.0)
    trace_sqrt = torch.sqrt(eigvals_middle).sum()

    # Squared Gaussian Frechet (2-Wasserstein) distance.
    distance = (
        mean_term
        + torch.trace(cov1)
        + torch.trace(cov2)
        - 2.0 * trace_sqrt
    )

    # Avoid tiny negative distances caused by floating-point roundoff.
    return max(float(distance.item()), 0.0)

# Step 12 - frechet_to_mixture
def frechet_to_mixture(x, mix):
    # Compute the empirical mean and unbiased sample covariance.
    sample_mean = x.mean(dim=0)
    sample_cov = torch.cov(x.T)

    means = mix["means"]      # Shape: (K, 2)
    stds = mix["stds"]        # Shape: (K,)
    weights = mix["weights"]  # Shape: (K,)

    # Compute the exact mean of the Gaussian mixture.
    mu = (weights.unsqueeze(-1) * means).sum(dim=0)

    # Compute the exact mixture covariance using the law of total covariance.
    eye = torch.eye(2, dtype=means.dtype, device=means.device)
    component_covs = (
        stds.square()[:, None, None] * eye
        + means.unsqueeze(-1) @ means.unsqueeze(-2)
    )

    Sigma = (
        weights[:, None, None] * component_covs
    ).sum(dim=0) - mu.unsqueeze(-1) @ mu.unsqueeze(-2)

    # Compare the sample statistics against the exact mixture moments.
    return gaussian_frechet_distance(
        sample_mean, sample_cov, mu, Sigma
    )


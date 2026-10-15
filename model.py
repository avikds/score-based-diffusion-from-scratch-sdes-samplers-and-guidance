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


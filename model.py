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


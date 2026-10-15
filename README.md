# Score-Based Diffusion from Scratch: SDEs, Samplers and Guidance

Build the score-based view of diffusion models in PyTorch on a 2-D Gaussian mixture whose true score is known, so every learned quantity can be checked against the exact one. Derive the analytic score of the data and of its noise-perturbed versions, train a noise-conditioned score network with denoising score matching (Song and Ermon, 2019), and draw samples with annealed Langevin dynamics, measuring quality with a Gaussian Frechet distance against the mixture's own moments. Then move to continuous time: the VP-SDE of Song et al. (2021), its marginal mean and standard deviation, the reverse-time SDE with an Euler-Maruyama predictor and a Langevin corrector, and predictor-corrector sampling. Turn the same score into deterministic samplers through the probability-flow ODE: Euler, DDIM with its eta knob that interpolates between the ODE path and ancestral sampling, and the second-order Heun sampler of Karras et al. (2022) that reaches the same quality with fewer network evaluations. Finish with conditional generation: a label-dropout conditional score network, classifier-free guidance at several scales, and classifier guidance with the mixture's exact Bayes classifier, measuring how guidance trades sample spread for class purity.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** sample_mixture
- [x] **2.** mixture_responsibilities
- [x] **3.** mixture_score

---

Built on Deep-ML.

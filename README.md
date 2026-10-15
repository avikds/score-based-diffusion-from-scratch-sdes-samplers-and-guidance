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
- [x] **4.** perturbed_mixture
- [x] **5.** dsm_loss
- [x] **6.** ScoreNet
- [x] **7.** train_score_net
- [x] **8.** geometric_sigmas
- [x] **9.** langevin_step
- [x] **10.** annealed_langevin
- [x] **11.** gaussian_frechet_distance
- [x] **12.** frechet_to_mixture
- [x] **13.** vp_sde_coefficients
- [x] **14.** vp_marginal
- [x] **15.** reverse_sde_predictor
- [x] **16.** langevin_corrector
- [x] **17.** pc_sampler
- [x] **18.** probability_flow_drift
- [x] **19.** ode_euler_sampler
- [x] **20.** ddim_sampler
- [x] **21.** heun_sampler
- [x] **22.** CondScoreNet
- [x] **23.** train_conditional_score_net

---

Built on Deep-ML.

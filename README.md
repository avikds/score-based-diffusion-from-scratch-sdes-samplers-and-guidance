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
- [x] **24.** cfg_score
- [x] **25.** classifier_guided_score
- [x] **26.** guidance_metrics

## Results

```
data: 3-component 2-D Gaussian mixture, means [[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]], stds [0.9, 0.9, 0.9], weights [0.3, 0.4, 0.3]
Frechet^2 of 3000 true samples to the exact mixture moments (sampling floor): 0.0012

=== Part 1: denoising score matching (Song and Ermon 2019) ===
VE score net: 10 geometric levels 5.00 .. 0.10, 600 Adam steps, DSM loss 0.7751 -> 0.6800
learned vs exact score of the perturbed mixture (relative L2 error over 3000 noisy points):
  sigma  0.1: relative error 0.770 | DSM loss learned 0.9905 vs at the exact score 0.9863 (irreducible)
  sigma  0.3: relative error 0.284 | DSM loss learned 0.9464 vs at the exact score 0.9464 (irreducible)
  sigma  1.0: relative error 0.190 | DSM loss learned 0.6595 vs at the exact score 0.6475 (irreducible)
  sigma  5.0: relative error 0.082 | DSM loss learned 0.1170 vs at the exact score 0.1107 (irreducible)

=== Part 2: annealed Langevin dynamics (Song and Ermon Algorithm 1: T=40 steps per level, eps=5e-3) ===
start N(0, sigma_1^2 I): Frechet^2 21.9396
  exact score  : Frechet^2 0.0049, fraction of samples per component [0.297, 0.403, 0.3] (true weights [0.3, 0.4, 0.3])
  learned score: Frechet^2 0.0159, fraction of samples per component [0.272, 0.434, 0.294] (true weights [0.3, 0.4, 0.3])

=== Part 3: VP-SDE (beta 0.1 -> 20, t_min 1e-3) and predictor-corrector sampling (Song et al. 2021) ===
marginal at t=1: alpha 0.0066, sigma 1.0000, so p_1 is N(0, I) to four decimals
VP score net: 600 Adam steps with t ~ U(t_min, 1], DSM loss 0.5821 -> 0.3509
learned vs exact score of p_t (relative L2 error): t=0.02: 0.946, t=0.1: 0.421, t=0.5: 0.043, t=1.0: 0.056  (the 1/sigma parametrisation amplifies a fixed eps-error at small t)
reverse-SDE sampling, Frechet^2 to the mixture, predictor-only vs predictor-corrector at equal NFE (snr 0.16):
  exact score  : P100C0 (NFE 100) 0.0019 | P50C1 (NFE 100) 0.0025
  learned score: P100C0 (NFE 100) 0.0113 | P50C1 (NFE 100) 0.0063
  (in 2-D with a near-exact score both are at the sampling floor; the corrector's gain in Song et al. is on images with larger score error)

=== Part 4: probability-flow ODE, DDIM and Heun (exact score, same prior noise for every sampler) ===
ODE-Euler vs DDIM(eta=0): mean endpoint distance from the same noise; DDIM(eta=1) vs the reverse-SDE Euler predictor: Frechet^2
  N= 10: |ODE - DDIM0| 0.2190 | Frechet^2 ODE 0.0054, DDIM0 0.0925 | DDIM1 0.1641, reverse-SDE Euler 0.0095
  N= 50: |ODE - DDIM0| 0.0480 | Frechet^2 ODE 0.0015, DDIM0 0.0075 | DDIM1 0.0096, reverse-SDE Euler 0.0032
  N=100: |ODE - DDIM0| 0.0243 | Frechet^2 ODE 0.0013, DDIM0 0.0035 | DDIM1 0.0037, reverse-SDE Euler 0.0034
Euler vs Heun at equal NFE: error = mean distance to a 200-step Heun reference (reference Frechet^2 0.0011)
  NFE  10: Euler  10 steps error 0.0752 Frechet^2 0.0054 | Heun   5 steps error 0.0988 Frechet^2 0.0102
  NFE  20: Euler  20 steps error 0.0347 Frechet^2 0.0024 | Heun  10 steps error 0.0269 Frechet^2 0.0025
  NFE  50: Euler  50 steps error 0.0132 Frechet^2 0.0015 | Heun  25 steps error 0.0044 Frechet^2 0.0013
  NFE 100: Euler 100 steps error 0.0065 Frechet^2 0.0013 | Heun  50 steps error 0.0011 Frechet^2 0.0012
same comparison with the learned VP score (Frechet^2): NFE 10: Euler 0.0191 Heun 0.0102 | NFE 20: Euler 0.0191 Heun 0.0130 | NFE 50: Euler 0.0184 Heun 0.0164 | NFE 100: Euler 0.0180 Heun 0.0172

=== Part 5: classifier-free guidance (Ho and Salimans 2022) and classifier guidance with the exact Bayes classifier ===
conditional score net, label dropout 0.2: 600 Adam steps, DSM loss 0.6265 -> 0.3025
at t=0.3: learned conditional score relative error 0.221, learned null-label (unconditional) score relative error 0.534
identity with exact scores: CFG at w=2 equals Bayes-classifier guidance at scale 3 (max abs difference 9.5e-07)
guidance sweep: accuracy of Bayes assignment to the intended class / within-class radius / shift of the class centre
  reference: unguided conditional samples have radius sqrt(2)*0.9 = 1.273 and shift 0; 'tilted' = the density p(x|y)^(1+w) p(x)^-w by grid integration
  w=0.0: learned CFG, Heun-25: acc 0.924 radius 1.507 shift 0.277 | learned CFG, PC-40: acc 0.947 radius 1.396 shift 0.071 | exact classifier guidance scale 1.0, Heun-25: acc 0.959 radius 1.270 shift 0.059 | tilted: radius 1.273 shift 0.000
  w=1.0: learned CFG, Heun-25: acc 1.000 radius 0.854 shift 1.318 | learned CFG, PC-40: acc 0.999 radius 1.078 shift 0.771 | exact classifier guidance scale 2.0, Heun-25: acc 1.000 radius 0.983 shift 0.735 | tilted: radius 1.208 shift 0.087
  w=2.0: learned CFG, Heun-25: acc 1.000 radius 0.629 shift 1.732 | learned CFG, PC-40: acc 1.000 radius 0.912 shift 1.131 | exact classifier guidance scale 3.0, Heun-25: acc 1.000 radius 0.894 shift 1.184 | tilted: radius 1.185 shift 0.124
  w=4.0: learned CFG, Heun-25: acc 1.000 radius 0.702 shift 1.973 | learned CFG, PC-40: acc 1.000 radius 0.731 shift 1.409 | exact classifier guidance scale 5.0, Heun-25: acc 1.000 radius 0.816 shift 1.826 | tilted: radius 1.161 shift 0.167
  (guidance raises purity to 100 percent and shrinks the within-class radius well below the unguided 1.27; the deterministic sampler also shifts the modes outward far more than the tilted density does, the stochastic PC sampler less so)

total time 9.7s
```

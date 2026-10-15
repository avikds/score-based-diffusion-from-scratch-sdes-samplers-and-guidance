"""
Score-Based Diffusion from Scratch: SDEs, Samplers and Guidance scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Score-Based Diffusion from Scratch: SDEs, Samplers and Guidance.

Reproduces, on a 2-D Gaussian mixture whose score is known exactly, the experiments of Song and Ermon (2019),
Song et al. (2021), Song, Meng and Ermon (DDIM, 2020), Karras et al. (2022) and Ho and Salimans (2022):
denoising score matching checked against the exact score; annealed Langevin dynamics; the VP-SDE with
predictor-corrector sampling; the probability-flow ODE with Euler, DDIM and Heun compared at equal numbers of
network evaluations; and classifier-free guidance against classifier guidance with the exact Bayes classifier,
measured by class purity, within-class radius and mode shift.
"""
import math
import time
import torch


def fmt(t, nd=3):
    return [round(v, nd) for v in torch.as_tensor(t).flatten().tolist()]


def tilted_density_moments(mix, w, half_width=8.0, n_grid=601):
    """(mean shift, radius) per class of the density p(x|y)^(1+w) p(x)^(-w) that guidance nominally targets, by grid integration."""
    g = torch.linspace(-half_width, half_width, n_grid)
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    means, stds, weights = mix["means"], mix["stds"], mix["weights"]

    def log_normal(k):
        return -((pts - means[k]) ** 2).sum(1) / (2 * stds[k] ** 2) - 2 * torch.log(stds[k]) - math.log(2 * math.pi)

    log_p = torch.logsumexp(torch.stack([torch.log(weights[k]) + log_normal(k) for k in range(len(stds))], 1), 1)
    out = []
    for k in range(len(stds)):
        p = torch.softmax((1 + w) * log_normal(k) - w * log_p, 0)
        centre = (p[:, None] * pts).sum(0)
        radius = float(torch.sqrt((p * ((pts - centre) ** 2).sum(1)).sum()))
        out.append((float((centre - means[k]).norm()), radius))
    return out


class ExactConditional:
    """The ideal CondScoreNet for a perturbed mixture: component score for a real label, mixture score for the null label."""

    def __init__(self, mix_t):
        self.mix_t = mix_t
        self.n_classes = len(mix_t["stds"])

    def __call__(self, x, sigma, y):
        k = y.clamp(max=self.n_classes - 1)
        cond = (self.mix_t["means"][k] - x) / self.mix_t["stds"][k][:, None] ** 2
        return torch.where((y == self.n_classes)[:, None], mixture_score(x, self.mix_t), cond)


def main():
    torch.manual_seed(0)
    t_start = time.time()
    mix = {"means": torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]]),
           "stds": torch.tensor([0.9, 0.9, 0.9]),
           "weights": torch.tensor([0.3, 0.4, 0.3])}
    n_eval = 3000
    x_data, y_data = sample_mixture(mix, n_eval)
    print(f"data: 3-component 2-D Gaussian mixture, means {mix['means'].tolist()}, stds {fmt(mix['stds'], 1)}, weights {fmt(mix['weights'], 1)}")
    print(f"Frechet^2 of {n_eval} true samples to the exact mixture moments (sampling floor): {frechet_to_mixture(x_data, mix):.4f}")

    # ---- Part 1: denoising score matching ----
    print("\n=== Part 1: denoising score matching (Song and Ermon 2019) ===")
    sigmas = geometric_sigmas(5.0, 0.1, 10)
    net_ve = ScoreNet(hidden=128)
    noise_ve = lambda n: (torch.ones(n), sigmas[torch.randint(len(sigmas), (n,))])
    losses = train_score_net(net_ve, mix, noise_ve, 600, 256, 3e-3)
    print(f"VE score net: {len(sigmas)} geometric levels {float(sigmas[0]):.2f} .. {float(sigmas[-1]):.2f}, 600 Adam steps, DSM loss {sum(losses[:20]) / 20:.4f} -> {sum(losses[-20:]) / 20:.4f}")
    analytic_ve = lambda x, sig: mixture_score(x, perturbed_mixture(mix, float(sig[0])))
    print(f"learned vs exact score of the perturbed mixture (relative L2 error over {n_eval} noisy points):")
    for s in (0.1, 0.3, 1.0, 5.0):
        sig = torch.full((n_eval,), s)
        eps = torch.randn_like(x_data)
        x_t = x_data + s * eps
        with torch.no_grad():
            learned = net_ve(x_t, sig)
            got = float(dsm_loss(net_ve, x_data, sig, eps))
        truth = analytic_ve(x_t, sig)
        rel = float((learned - truth).norm(dim=1).mean() / truth.norm(dim=1).mean())
        floor = float(dsm_loss(analytic_ve, x_data, sig, eps))
        print(f"  sigma {s:4.1f}: relative error {rel:.3f} | DSM loss learned {got:.4f} vs at the exact score {floor:.4f} (irreducible)")

    # ---- Part 2: annealed Langevin dynamics ----
    print("\n=== Part 2: annealed Langevin dynamics (Song and Ermon Algorithm 1: T=40 steps per level, eps=5e-3) ===")
    torch.manual_seed(1)
    x_init = torch.randn(n_eval, 2) * float(sigmas[0])
    print(f"start N(0, sigma_1^2 I): Frechet^2 {frechet_to_mixture(x_init, mix):.4f}")
    for name, fn in (("exact score", analytic_ve), ("learned score", lambda x, sig: net_ve(x, sig))):
        torch.manual_seed(2)
        with torch.no_grad():
            xs = annealed_langevin(fn, x_init, sigmas, 40, 5e-3)
        frac = mixture_responsibilities(xs, mix).argmax(1).bincount(minlength=3).float() / n_eval
        print(f"  {name:13s}: Frechet^2 {frechet_to_mixture(xs, mix):.4f}, fraction of samples per component {fmt(frac)} (true weights {fmt(mix['weights'], 1)})")

    # ---- Part 3: VP-SDE and predictor-corrector ----
    print("\n=== Part 3: VP-SDE (beta 0.1 -> 20, t_min 1e-3) and predictor-corrector sampling (Song et al. 2021) ===")
    bmin, bmax, tmin = 0.1, 20.0, 1e-3
    a1, s1 = vp_marginal(torch.tensor([1.0]), bmin, bmax)
    print(f"marginal at t=1: alpha {float(a1):.4f}, sigma {float(s1):.4f}, so p_1 is N(0, I) to four decimals")
    net_vp = ScoreNet(hidden=128)
    noise_vp = lambda n: vp_marginal(tmin + (1 - tmin) * torch.rand(n), bmin, bmax)
    torch.manual_seed(3)
    losses = train_score_net(net_vp, mix, noise_vp, 600, 256, 3e-3)
    print(f"VP score net: 600 Adam steps with t ~ U(t_min, 1], DSM loss {sum(losses[:20]) / 20:.4f} -> {sum(losses[-20:]) / 20:.4f}")

    def analytic_vp(x, t):
        a, s = vp_marginal(t[:1], bmin, bmax)
        return mixture_score(x, perturbed_mixture(mix, float(s[0]), float(a[0])))

    learned_vp = lambda x, t: net_vp(x, vp_marginal(t, bmin, bmax)[1])
    parts = []
    for t in (0.02, 0.1, 0.5, 1.0):
        tt = torch.full((n_eval,), t)
        a, s = vp_marginal(tt, bmin, bmax)
        x_t = a[:, None] * x_data + s[:, None] * torch.randn_like(x_data)
        with torch.no_grad():
            rel = float((learned_vp(x_t, tt) - analytic_vp(x_t, tt)).norm(dim=1).mean() / analytic_vp(x_t, tt).norm(dim=1).mean())
        parts.append(f"t={t}: {rel:.3f}")
    print("learned vs exact score of p_t (relative L2 error): " + ", ".join(parts) + "  (the 1/sigma parametrisation amplifies a fixed eps-error at small t)")
    print("reverse-SDE sampling, Frechet^2 to the mixture, predictor-only vs predictor-corrector at equal NFE (snr 0.16):")
    for name, fn in (("exact score", analytic_vp), ("learned score", learned_vp)):
        row = []
        for n_pred, n_corr in ((100, 0), (50, 1)):
            torch.manual_seed(4)
            with torch.no_grad():
                xs = pc_sampler(fn, torch.randn(n_eval, 2), n_pred, bmin, bmax, 0.16, n_corr, tmin)
            row.append(f"P{n_pred}C{n_corr} (NFE {n_pred * (1 + n_corr)}) {frechet_to_mixture(xs, mix):.4f}")
        print(f"  {name:13s}: " + " | ".join(row))
    print("  (in 2-D with a near-exact score both are at the sampling floor; the corrector's gain in Song et al. is on images with larger score error)")

    # ---- Part 4: deterministic samplers ----
    print("\n=== Part 4: probability-flow ODE, DDIM and Heun (exact score, same prior noise for every sampler) ===")
    torch.manual_seed(5)
    x_init = torch.randn(n_eval, 2)
    print("ODE-Euler vs DDIM(eta=0): mean endpoint distance from the same noise; DDIM(eta=1) vs the reverse-SDE Euler predictor: Frechet^2")
    for n_steps in (10, 50, 100):
        with torch.no_grad():
            x_ode = ode_euler_sampler(analytic_vp, x_init, n_steps, bmin, bmax, tmin)
            x_ddim0 = ddim_sampler(analytic_vp, x_init, n_steps, bmin, bmax, 0.0, tmin)
            torch.manual_seed(6)
            x_ddim1 = ddim_sampler(analytic_vp, x_init, n_steps, bmin, bmax, 1.0, tmin)
            torch.manual_seed(6)
            x_anc = pc_sampler(analytic_vp, x_init, n_steps, bmin, bmax, 0.16, 0, tmin)
        print(f"  N={n_steps:3d}: |ODE - DDIM0| {float((x_ode - x_ddim0).norm(dim=1).mean()):.4f} | Frechet^2 ODE {frechet_to_mixture(x_ode, mix):.4f}, DDIM0 {frechet_to_mixture(x_ddim0, mix):.4f} | DDIM1 {frechet_to_mixture(x_ddim1, mix):.4f}, reverse-SDE Euler {frechet_to_mixture(x_anc, mix):.4f}")
    with torch.no_grad():
        x_ref = heun_sampler(analytic_vp, x_init, 200, bmin, bmax, tmin)
    print(f"Euler vs Heun at equal NFE: error = mean distance to a 200-step Heun reference (reference Frechet^2 {frechet_to_mixture(x_ref, mix):.4f})")
    for nfe in (10, 20, 50, 100):
        with torch.no_grad():
            x_e = ode_euler_sampler(analytic_vp, x_init, nfe, bmin, bmax, tmin)
            x_h = heun_sampler(analytic_vp, x_init, nfe // 2, bmin, bmax, tmin)
        print(f"  NFE {nfe:3d}: Euler {nfe:3d} steps error {float((x_e - x_ref).norm(dim=1).mean()):.4f} Frechet^2 {frechet_to_mixture(x_e, mix):.4f} | Heun {nfe // 2:3d} steps error {float((x_h - x_ref).norm(dim=1).mean()):.4f} Frechet^2 {frechet_to_mixture(x_h, mix):.4f}")
    parts = []
    for nfe in (10, 20, 50, 100):
        with torch.no_grad():
            x_e = ode_euler_sampler(learned_vp, x_init, nfe, bmin, bmax, tmin)
            x_h = heun_sampler(learned_vp, x_init, nfe // 2, bmin, bmax, tmin)
        parts.append(f"NFE {nfe}: Euler {frechet_to_mixture(x_e, mix):.4f} Heun {frechet_to_mixture(x_h, mix):.4f}")
    print("same comparison with the learned VP score (Frechet^2): " + " | ".join(parts))

    # ---- Part 5: guidance ----
    print("\n=== Part 5: classifier-free guidance (Ho and Salimans 2022) and classifier guidance with the exact Bayes classifier ===")
    torch.manual_seed(7)
    cnet = CondScoreNet(n_classes=3, hidden=128)
    losses = train_conditional_score_net(cnet, mix, noise_vp, 600, 256, 3e-3, 0.2)
    print(f"conditional score net, label dropout 0.2: 600 Adam steps, DSM loss {sum(losses[:20]) / 20:.4f} -> {sum(losses[-20:]) / 20:.4f}")
    n_per = 600
    y_q = torch.arange(3).repeat_interleave(n_per)
    torch.manual_seed(8)
    x_init = torch.randn(3 * n_per, 2)
    t_chk = torch.full((3 * n_per,), 0.3)
    a, s = vp_marginal(t_chk, bmin, bmax)
    mix_t = perturbed_mixture(mix, float(s[0]), float(a[0]))
    x_t = a[:, None] * x_data[: 3 * n_per] + s[:, None] * torch.randn(3 * n_per, 2)
    with torch.no_grad():
        rel_c = float((cnet(x_t, s, y_q) - ExactConditional(mix_t)(x_t, s, y_q)).norm(dim=1).mean() / ExactConditional(mix_t)(x_t, s, y_q).norm(dim=1).mean())
        null = torch.full_like(y_q, 3)
        rel_u = float((cnet(x_t, s, null) - mixture_score(x_t, mix_t)).norm(dim=1).mean() / mixture_score(x_t, mix_t).norm(dim=1).mean())
    print(f"at t=0.3: learned conditional score relative error {rel_c:.3f}, learned null-label (unconditional) score relative error {rel_u:.3f}")
    gap = float((cfg_score(ExactConditional(mix_t), x_init, s, y_q, 2.0) - classifier_guided_score(x_init, mix_t, y_q, 3.0)).abs().max())
    print(f"identity with exact scores: CFG at w=2 equals Bayes-classifier guidance at scale 3 (max abs difference {gap:.1e})")
    print("guidance sweep: accuracy of Bayes assignment to the intended class / within-class radius / shift of the class centre")
    print("  reference: unguided conditional samples have radius sqrt(2)*0.9 = 1.273 and shift 0; 'tilted' = the density p(x|y)^(1+w) p(x)^-w by grid integration")
    for w in (0.0, 1.0, 2.0, 4.0):
        cfg_fn = lambda x, t: cfg_score(cnet, x, vp_marginal(t, bmin, bmax)[1], y_q, w)

        def cls_fn(x, t):
            a_t, s_t = vp_marginal(t[:1], bmin, bmax)
            return classifier_guided_score(x, perturbed_mixture(mix, float(s_t[0]), float(a_t[0])), y_q, 1.0 + w)

        with torch.no_grad():
            m_cfg_ode = guidance_metrics(heun_sampler(cfg_fn, x_init, 25, bmin, bmax, tmin), y_q, mix)
            torch.manual_seed(9)
            m_cfg_pc = guidance_metrics(pc_sampler(cfg_fn, x_init, 40, bmin, bmax, 0.16, 1, tmin), y_q, mix)
            m_cls_ode = guidance_metrics(heun_sampler(cls_fn, x_init, 25, bmin, bmax, tmin), y_q, mix)
        tilt = tilted_density_moments(mix, w)
        tilt_shift = sum(m[0] for m in tilt) / 3
        tilt_rad = sum(m[1] for m in tilt) / 3
        print(f"  w={w:3.1f}: learned CFG, Heun-25: acc {m_cfg_ode[0]:.3f} radius {m_cfg_ode[1]:.3f} shift {m_cfg_ode[2]:.3f} | learned CFG, PC-40: acc {m_cfg_pc[0]:.3f} radius {m_cfg_pc[1]:.3f} shift {m_cfg_pc[2]:.3f} | exact classifier guidance scale {1 + w:.1f}, Heun-25: acc {m_cls_ode[0]:.3f} radius {m_cls_ode[1]:.3f} shift {m_cls_ode[2]:.3f} | tilted: radius {tilt_rad:.3f} shift {tilt_shift:.3f}")
    print("  (guidance raises purity to 100 percent and shrinks the within-class radius well below the unguided 1.27; the deterministic sampler also shifts the modes outward far more than the tilted density does, the stochastic PC sampler less so)")
    print(f"\ntotal time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()


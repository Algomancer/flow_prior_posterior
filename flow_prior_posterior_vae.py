import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import distributions as D, Tensor
from tqdm.auto import trange


def sum_except_batch(x):
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)


def bounded_exp_scale(raw_scale, scale_factor):
    log_s = torch.tanh(raw_scale) * scale_factor       # symmetric, bounded
    s = torch.exp(log_s)
    return s, log_s


class Backbone(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, model_dim)
        self.fc2 = nn.Linear(model_dim, model_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, offsets=None, lengths=None):
        h = F.silu(self.fc1(x))
        return x + self.fc2(h)


class AffineCoupling(nn.Module):
    def __init__(self, input_dim, model_dim, cond_dim, scale_factor=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.d_a = input_dim // 2
        self.d_b = input_dim - self.d_a

        self.params = nn.Linear(model_dim, self.d_b * 2)
        self.input_proj = nn.Linear(self.d_a + cond_dim, model_dim)

        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.params.weight)
        nn.init.zeros_(self.params.bias)

        self.scale_factor = nn.Parameter(torch.tensor(scale_factor), requires_grad=False)

    def embed(self, x, conditioning=None):
        x_a, _ = x.split([self.d_a, self.d_b], dim=-1)
        if conditioning is None:
            inp = x_a
        else:
            inp = torch.cat([x_a, conditioning], dim=-1)
        return self.input_proj(inp)

    def forward(self, x, context):
        x_a, x_b = x.split([self.d_a, self.d_b], dim=-1)
        params = self.params(context)           # (B, 2 * d_b)
        bias, raw_scale = params.chunk(2, dim=-1)

        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        log_det = sum_except_batch(log_s.float())

        z_b = (x_b + bias) * scale
        z = torch.cat([x_a, z_b], dim=-1)
        return z, log_det

    def inverse(self, z, context):
        z_a, z_b = z.split([self.d_a, self.d_b], dim=-1)
        params = self.params(context)           # (B, 2 * d_b)
        bias, raw_scale = params.chunk(2, dim=-1)

        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        log_det = sum_except_batch(log_s.float())

        x_b = (z_b / scale) - bias
        x = torch.cat([z_a, x_b], dim=-1)
        return x, -log_det


class Coupling(nn.Module):
    def __init__(self, input_dim, model_dim, cond_dim):
        super().__init__()
        self.input_dim = input_dim
        self.coupling = AffineCoupling(input_dim, model_dim, cond_dim)
        self.backbone = Backbone(model_dim, model_dim)

    def forward(self, x, conditioning=None):
        embed = self.coupling.embed(x, conditioning)
        context = self.backbone(embed)
        z, log_det = self.coupling(x, context)
        return z, log_det

    def inverse(self, z, conditioning=None):
        embed = self.coupling.embed(z, conditioning)
        context = self.backbone(embed)
        x, log_det = self.coupling.inverse(z, context)
        return x, log_det


class ReversePermute(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        indices = torch.arange(input_dim - 1, -1, -1)
        self.register_buffer("indices", indices)
        self.register_buffer("inverse_indices", torch.argsort(indices))

    def forward(self, x, conditioning=None):
        z = torch.index_select(x, -1, self.indices)
        log_det = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        return z, log_det

    def inverse(self, z, conditioning=None):
        x = torch.index_select(z, -1, self.inverse_indices)
        log_det = torch.zeros(z.size(0), dtype=torch.float32, device=z.device)
        return x, log_det


class FlowSequence(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = flows

    def forward(self, x, conditioning=None):
        log_det = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        for flow in self.flows:
            x, ldj = flow(x, conditioning)
            log_det += ldj
        return x, log_det

    def inverse(self, x, conditioning=None):
        log_det = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        for flow in reversed(self.flows):
            x, ldj = flow.inverse(x, conditioning)
            log_det += ldj
        return x, log_det


def build_flow(num_flows, latent_dim, model_dim, cond_dim):
    flows = []
    for _ in range(num_flows):
        flows.append(Coupling(latent_dim, model_dim, cond_dim))
        flows.append(ReversePermute(latent_dim))
    return FlowSequence(nn.ModuleList(flows))


def gen_data(n):
    scale = 4.
    centers = torch.tensor([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1. / np.sqrt(2), 1. / np.sqrt(2)],
        [1. / np.sqrt(2), -1. / np.sqrt(2)],
        [-1. / np.sqrt(2), 1. / np.sqrt(2)],
        [-1. / np.sqrt(2), -1. / np.sqrt(2)]
    ], dtype=torch.float32)
    centers = scale * centers

    x = torch.randn(n, 2)
    x = 0.5 * x

    center_ids = torch.randint(0, 8, (n,))
    x = x + centers[center_ids]

    x = x / 2 ** 0.5
    return x


class Encoder(nn.Module):
    def __init__(self, input_dim, model_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, latent_dim * 2)
        )


    def forward(self, x):
        stats = self.net(x)
        mu, log_std = stats.chunk(2, dim=-1)
        std = torch.exp(log_std)
        return D.Normal(mu, std), mu


class PriorObservation(nn.Module):
    def __init__(self, latent_size: int, data_size: int, noise_std: float, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_size),
        )

        self.noise_std = noise_std

    def get_coeffs(self, z: Tensor) -> tuple[Tensor, Tensor]:
        m = self.net(z)
        s = torch.ones_like(m) * self.noise_std
        return m, s

    def forward(self, z: Tensor) -> D.Distribution:
        m, s = self.get_coeffs(z)
        return D.Independent(D.Normal(m, s), 1)


class VAEFlowPosterior(nn.Module):
    def __init__(self, input_dim, model_dim, latent_dim,
                 num_flows, gamma=50.0, num_prior_flows=0):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.gamma = gamma  # ≈ 1 / σ²

        self.encoder = Encoder(input_dim, model_dim, latent_dim)

        noise_std = (1.0 / gamma) ** 0.5
        self.obs_model = PriorObservation(latent_dim, input_dim, noise_std)

        # posterior flow: q(z|x)
        self.flow = build_flow(num_flows, latent_dim, model_dim, cond_dim=latent_dim)

        # prior flow: p(z), unconditional
        self.prior_flow = (
            build_flow(num_prior_flows, latent_dim, model_dim, cond_dim=0)
            if num_prior_flows > 0 else None
        )

    @torch.compile()
    def forward(self, x, beta=None):
        # base posterior q0(u|x) and conditioning features
        q0, h = self.encoder(x)
        u = q0.rsample()

        # posterior flow: z = f(u; h)
        z, log_det_post = self.flow(u, conditioning=h)

        # log q(z|x) = log q0(u|x) - log|det ∂z/∂u|
        log_q0 = q0.log_prob(u).sum(-1)
        log_q_z = log_q0 - log_det_post

        # prior p(z) with optional flow
        if self.prior_flow is None:
            p_z = D.Normal(torch.zeros_like(z), torch.ones_like(z))
            log_p_z = p_z.log_prob(z).sum(-1)
        else:
            # invert prior flow: z -> ε, log_det_inv = log|det ∂ε/∂z|
            eps, log_det_inv = self.prior_flow.inverse(z, conditioning=None)
            p_eps = D.Normal(torch.zeros_like(eps), torch.ones_like(eps))
            log_p_eps = p_eps.log_prob(eps).sum(-1)
            log_p_z = log_p_eps + log_det_inv

        # observation model p(x|z)
        px_given_z = self.obs_model(z)
        log_p_x_given_z = px_given_z.log_prob(x)



        # ELBO = log p(x|z) + log p(z) - log q(z|x)
        elbo = log_p_x_given_z + (log_p_z - log_q_z) * beta
        loss = -elbo.mean()

        kl = (log_q_z - log_p_z).mean()

        stats = {
            "loss": loss.detach(),
            "elbo": elbo.mean().detach(),
            "recon_logprob": log_p_x_given_z.mean().detach(),
            "kl": kl.detach(),
        }
        return loss, stats

    def reconstruct(self, x):
        q0, h = self.encoder(x)
        u = q0.rsample()
        z, _ = self.flow(u, conditioning=h)
        px = self.obs_model(z)
        x_hat = px.mean
        return x_hat

    def sample(self, n, device=None, refine_steps=10, refine_lr=0.1):
        if device is None:
            device = next(self.parameters()).device

        # Sample from prior (z or prior_flow)
        if self.prior_flow is None:
            z = torch.randn(n, self.latent_dim, device=device)
        else:
            eps = torch.randn(n, self.latent_dim, device=device)
            z, _ = self.prior_flow(eps, conditioning=None)

        # Refine z with gradient ascent on likelihood under the prior, anchored to z
        z = z.detach().clone()
        z.requires_grad_(True)
        for _ in range(refine_steps):
            # likelihood under the prior: p(z) (standard normal or with prior_flow)
            if self.prior_flow is None:
                log_prob = -0.5 * ((z ** 2).sum(-1) + z.shape[-1] * np.log(2 * np.pi))
            else:
                # Map z back to eps, compute base prior log_prob and adjust by log-det
                eps, log_det = self.prior_flow.inverse(z, conditioning=None)
                log_p_eps = -0.5 * ((eps ** 2).sum(-1) + z.shape[-1] * np.log(2 * np.pi))
                log_prob = log_p_eps + log_det
            grad = torch.autograd.grad(log_prob.mean(), z, create_graph=False)[0]
            z = z + refine_lr * grad
            z = z.detach()
            z.requires_grad_(True)
        z = z.detach()

        px = self.obs_model(z)
        x_samples = px.rsample()
        return z, x_samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)
    np.random.seed(0)

    input_dim = 2
    model_dim = 256
    latent_dim = 2
    num_flows = 4
    num_prior_flows = 4

    batch_size = 256
    num_steps = 10000
    lr = 1e-3
    gamma = 50.0   # ≈ 1 / σ² for observation noise

    model = VAEFlowPosterior(
        input_dim,
        model_dim,
        latent_dim,
        num_flows,
        gamma=gamma,
        num_prior_flows=num_prior_flows,
    ).to(device)

    warmup_steps = 1000

    # Set initial learning rate for optimizer to zero
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = trange(num_steps)
    for step in pbar:
        # Manually adjust the learning rate for linear warmup
        if step < warmup_steps:
            warmup_lr = lr * (step + 1) / warmup_steps
        else:
            warmup_lr = lr
        for param_group in optim.param_groups:
            param_group['lr'] = warmup_lr

        model.train()
        x = gen_data(batch_size).to(device)

        loss, stats = model(x, beta=1.0)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step + 1) % 100 == 0:
            pbar.set_postfix(
                loss=float(stats["loss"].cpu()),
                recon=float(stats["recon_logprob"].cpu()),
                kl=float(stats["kl"].cpu()),
                elbo=float(stats["elbo"].cpu()),
            )

            model.eval()
            # recon / data
            x_eval = gen_data(1024).to(device)
            x_recon = model.reconstruct(x_eval)

            # samples from flow prior
            _, x_samples = model.sample(1024, device=device)

            x_eval_np = x_eval.detach().cpu().numpy()
            x_recon_np = x_recon.detach().cpu().numpy()
            x_samples_np = x_samples.detach().cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            scatter_kwargs = dict(s=5, alpha=0.6)

            axs[0].scatter(x_eval_np[:, 0], x_eval_np[:, 1], **scatter_kwargs)
            axs[0].set_title("Data")

            axs[1].scatter(x_recon_np[:, 0], x_recon_np[:, 1], **scatter_kwargs)
            axs[1].set_title("Reconstructions")

            axs[2].scatter(x_samples_np[:, 0], x_samples_np[:, 1], **scatter_kwargs)
            axs[2].set_title("Samples")

            for ax in axs:
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_aspect("equal")

            plt.tight_layout()
            plt.savefig("results.jpg")
            plt.close(fig)

# flow_prior_posterior
A clean example of a conditional flow posterior and a prior flow based vae as well as prior based iterative refinement.

Just because I get asked this question pretty often when talking about vaes with flexible base distributions and figured i'd make a simple example. 

If we think of diffusion models as hierachical VAE models, this is the vae analogy for neural flow diffusion with a learned forward mode and backwards mode. 

Essentially you define your posterior and prior as flexible distributions. With the posterior flow being conditioned on the deterministic mean, such that you can tighten your posterior distribution avoiding posterior collapse in a lot of cases, since the flow can move the random variable to have tighter decoder dependent geometry.


I also added an example of gradient ascent, you can imagine if you had a reward model trained on z's, you could do inference time constrained search to find better z's to decode from. I did this with early Llama models a few years back.

```python
    def sample(self, n, device=None, refine_steps=10, refine_lr=0.1, lambda_=0.1):
        if device is None:
            device = next(self.parameters()).device

        # Sample from prior (z or prior_flow)
        if self.prior_flow is None:
            z = torch.randn(n, self.latent_dim, device=device)
        else:
            eps = torch.randn(n, self.latent_dim, device=device)
            z, _ = self.prior_flow(eps, conditioning=None)

        z0 = z.detach().clone()
        z = z0.clone()
        z.requires_grad_(True)
        # This is where you would add a reward model that is trained on z's, and gradient ascent to better samples under the rewards.
        for _ in range(refine_steps):
            if self.prior_flow is None:
                log_p = -0.5 * (z ** 2).sum(-1)
            else:
                eps, log_det = self.prior_flow.inverse(z, conditioning=None)
                log_p_eps = -0.5 * (eps ** 2).sum(-1)
                log_p = log_p_eps + log_det

            anchor = -lambda_ * ((z - z0) ** 2).sum(-1)
            obj = (log_p + anchor).mean()

            grad = torch.autograd.grad(obj, z, create_graph=False)[0]
            z = z + refine_lr * grad
            z = z.detach()
            z.requires_grad_(True)
        z = z.detach()

        px = self.obs_model(z)
        x_samples = px.rsample()
        return z, x_samples
```


```
@misc{algomancer2025,
  author = {@algomancer},
  title  = {Some Dumb Shit},
  year   = {2025}
}
```

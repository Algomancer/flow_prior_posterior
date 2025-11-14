# flow_prior_posterior
A clean example of a conditional flow posterior and a prior flow based vae as well as prior based iterative refinement.

Just because I get asked this question pretty often when talking about vaes with flexible base distributions and figured i'd make a simple example. 

If we think of diffusion models as hierachical diffusion models, this is the vae analogy for neural flow diffusion with a learned forward mode and backwards mode. 

Essentially you define your posterior and prior as flexible distributions. With the posterior flow being conditioned on the deterministic mean, such that you can tighten your posterior distribution avoiding posterior collapse in a lot of cases, since the flow can move the random variable to have tighter decoder dependent geometry.


```
@misc{algomancer2025,
  author = {@algomancer},
  title  = {Some Dumb Shit},
  year   = {2025}
}
```

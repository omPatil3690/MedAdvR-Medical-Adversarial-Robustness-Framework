22. One important issue in this implementation
    There is a conceptual issue worth being aware of.
    Your comments say:
    """Specialized loss for edge generator: -|∇(x_adv) - ∇x|"""

and similarly for intensity and texture.
So these losses deliberately encourage large perturbations.
But your generator architecture already has:
delta = torch.tanh(self.out(x))return img + self.eps \* delta

with:
eps = 0.05

Therefore the perturbation is bounded approximately by:
\[
-0.05 \leq \delta \leq 0.05
\]
assuming images are in [0,1].
So you have two mechanisms controlling perturbation size:
Generator:
tanh + ε
↓
hard perturbation bound

Realism loss:
SSIM + specialized realism
↓
soft realism constraint

That's a sensible adversarial-design idea.

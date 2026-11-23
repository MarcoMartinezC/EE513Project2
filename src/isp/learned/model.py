"""DnCNN-style residual denoiser for packed Bayer (4 channels).

Differences from the exercise's ``DenoiseNet`` and why:

* Input is packed RGGB (4 planes at half resolution), the same place in the
  pipeline as the classical guided filter, so the comparison is like for like
  and the noise is still the independent Poisson-Gaussian noise you measured.
* An optional noise-level map (predicted std per pixel from your noise model)
  is concatenated to the input, so one network can serve several ISOs. Without
  it, a network trained at one noise level is tuned to that level only, which
  is one of the failure modes the spec asks you to find.
* Default initialisation everywhere. Starting the last layer at or near zero
  (so the untrained net is the identity) looks attractive but, measured on the
  synthetic camera, either slowed training badly (zero) or let the last ReLU
  layer die on one seed out of two (std 1e-3), after which the network outputs
  its input forever. ``train`` checks for that and warns.
"""

from __future__ import annotations

import torch
from torch import nn


class DnCNN(nn.Module):
    def __init__(self, in_ch: int = 4, width: int = 32, depth: int = 8, noise_map: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.noise_map = noise_map
        c0 = in_ch * 2 if noise_map else in_ch
        layers: list[nn.Module] = [nn.Conv2d(c0, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, in_ch, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, 4, h, w) normalised packed Bayer. sigma: (B, 4, h, w) predicted noise std."""
        if self.noise_map:
            if sigma is None:
                raise ValueError("this model was built with noise_map=True; pass sigma")
            inp = torch.cat([x, sigma], dim=1)
        else:
            inp = x
        return x - self.body(inp)

    @torch.no_grad()
    def alive_fraction(self, x: torch.Tensor, sigma: torch.Tensor | None = None) -> list[float]:
        """Fraction of positive activations after each ReLU. Near 0 in any layer = dead layer."""
        h = torch.cat([x, sigma], 1) if self.noise_map else x
        out = []
        for layer in self.body:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                out.append(float((h > 0).float().mean()))
        return out

    def config(self) -> dict:
        convs = [m for m in self.body if isinstance(m, nn.Conv2d)]
        return {"in_ch": self.in_ch, "width": convs[0].out_channels, "depth": len(convs),
                "noise_map": self.noise_map}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def conv_params(c_in: int, c_out: int, k: int = 3, bias: bool = True) -> int:
    """Hand formula for one conv layer, to check against PyTorch (as in the on-ramp exercise)."""
    return c_out * (c_in * k * k + (1 if bias else 0))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

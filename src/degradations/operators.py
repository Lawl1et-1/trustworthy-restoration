"""Linear forward operators on real images, with exact range/null-space projectors.

Single-coil Cartesian MRI is simulated from magnitude images: A x = M F x with an orthonormal FFT.
Because a real image has Hermitian k-space, masks are made conjugate-symmetric (see masks.py), so
A^T A = Re(F^H M F) is an exact orthogonal projector on real images. Inpainting is A x = m * x.
"""
import torch


class LinearOp:
    """kind in {"mri", "inpaint"}; mask has shape (B, 1, H, W) or (H, W), values in {0, 1}."""

    def __init__(self, kind, mask):
        assert kind in ("mri", "inpaint")
        self.kind = kind
        self.mask = mask.float()

    def A(self, x):
        if self.kind == "mri":
            return torch.fft.fft2(x, norm="ortho") * self.mask
        return x * self.mask

    def AT(self, y):
        """Adjoint w.r.t. the real inner product <u, v> = Re sum conj(u) v; equals the pseudo-inverse."""
        if self.kind == "mri":
            return torch.fft.ifft2(y * self.mask, norm="ortho").real
        return y * self.mask

    def P_R(self, x):
        return self.AT(self.A(x))

    def P_N(self, x):
        return x - self.P_R(x)

    def data_consistency(self, x, y):
        """Hard DC: keep the null-space part of x, replace the range part by A^+ y."""
        return self.P_N(x) + self.AT(y)

    def range_noise_var(self, sigma):
        """Per-pixel variance of A^+ n for the noise model in degrade.py (sigma = image-domain std)."""
        m = self.mask
        if self.kind == "mri":
            frac = m.flatten(-2).mean(-1)[..., None, None]  # |S| / N
            return (sigma ** 2) * frac * torch.ones_like(m)
        return (sigma ** 2) * m

"""
glhcm_loss.py  -  differentiable GLHCM texture loss for PC-MambaGAN

Replaces the pyradiomics/NumPy GLHCM computation in i2i_mamba_one.py, which
was detached from the graph (.detach().cpu().numpy()) and therefore produced
no gradient. Every step below is a PyTorch operation, so the loss trains the
generator.

Pipeline (per image, batch-wise on GPU):
  1. Hessian with Gaussian-derivative filters (sigma = 2)
  2. Two response maps: Frobenius norm and smallest eigenvalue
  3. Map both images to [0, 1] using the REFERENCE image's min/max
  4. Soft quantization into N_LEVELS gray levels (Gaussian memberships)
  5. Soft co-occurrence matrix: distance 1, angles 0/45/90/135 deg,
     symmetric, averaged over angles, normalized to sum 1
  6. Seven descriptors: contrast, joint energy, autocorrelation,
     joint entropy, IDM, cluster shade, cluster tendency
  7. Relative L1 difference, averaged over descriptors and both maps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLHCMLoss(nn.Module):
    def __init__(self, sigma=2.0, n_levels=32, eps=1e-8):
        super().__init__()
        self.n_levels = n_levels
        self.eps = eps

        # Gaussian derivative kernels (separable), radius = 3 sigma
        r = int(math.ceil(3 * sigma))
        x = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        g1 = -x / sigma ** 2 * g                       # first derivative
        g2 = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g  # second derivative
        g2 = g2 - g2.mean()                             # zero response to constants
        self.register_buffer("g0", g.view(1, 1, 1, -1))
        self.register_buffer("g1", g1.view(1, 1, 1, -1))
        self.register_buffer("g2", g2.view(1, 1, 1, -1))
        self.pad = r

        # bin centres and membership width for soft quantization
        self.register_buffer("centres", torch.linspace(0.0, 1.0, n_levels).view(1, n_levels, 1, 1))
        self.width = 1.0 / (n_levels - 1)

        # gray-level index grids for the descriptors
        lv = torch.arange(n_levels, dtype=torch.float32)
        self.register_buffer("p_idx", lv.view(1, -1, 1))
        self.register_buffer("q_idx", lv.view(1, 1, -1))

    # --- filtering helpers -------------------------------------------------
    def _sep(self, img, ky, kx):
        img = F.pad(img, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        img = F.conv2d(img, kx)                          # along width
        img = F.conv2d(img, ky.transpose(2, 3))          # along height
        return img

    def _hessian_maps(self, img):
        ixx = self._sep(img, self.g0, self.g2)
        iyy = self._sep(img, self.g2, self.g0)
        ixy = self._sep(img, self.g1, self.g1)
        frob = torch.sqrt(ixx ** 2 + 2 * ixy ** 2 + iyy ** 2 + self.eps)
        half_diff = torch.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2 + self.eps)
        min_eig = (ixx + iyy) / 2 - half_diff
        return frob, min_eig

    # --- soft GLCM ----------------------------------------------------------
    def _memberships(self, x):
        # x: (B,1,H,W) in [0,1] -> (B,L,H,W), sums to 1 over L
        m = torch.exp(-((x - self.centres) ** 2) / (2 * self.width ** 2))
        return m / (m.sum(dim=1, keepdim=True) + self.eps)

    def _glcm(self, m):
        # m: (B,L,H,W); offsets for 0, 45, 90, 135 degrees at distance 1
        B, L, H, W = m.shape
        pairs = [
            (m[:, :, :, :-1], m[:, :, :, 1:]),       # 0 deg
            (m[:, :, 1:, :-1], m[:, :, :-1, 1:]),    # 45 deg
            (m[:, :, :-1, :], m[:, :, 1:, :]),       # 90 deg
            (m[:, :, :-1, :-1], m[:, :, 1:, 1:]),    # 135 deg
        ]
        P = 0
        for a, b in pairs:
            c = torch.einsum("bpn,bqn->bpq", a.reshape(B, L, -1), b.reshape(B, L, -1))
            c = c + c.transpose(1, 2)                  # symmetric
            P = P + c / (c.sum(dim=(1, 2), keepdim=True) + self.eps)
        return P / len(pairs)                          # (B,L,L), sums to 1

    def _descriptors(self, P):
        p, q = self.p_idx, self.q_idx
        mu_x = (p * P).sum(dim=(1, 2), keepdim=True)
        mu_y = (q * P).sum(dim=(1, 2), keepdim=True)
        s = p + q - mu_x - mu_y
        feats = [
            ((p - q) ** 2 * P).sum(dim=(1, 2)),                    # contrast
            (P ** 2).sum(dim=(1, 2)),                              # joint energy
            (p * q * P).sum(dim=(1, 2)),                           # autocorrelation
            -(P * torch.log(P + self.eps)).sum(dim=(1, 2)),        # joint entropy
            (P / (1 + (p - q) ** 2)).sum(dim=(1, 2)),              # IDM
            (s ** 3 * P).sum(dim=(1, 2)),                          # cluster shade
            (s ** 2 * P).sum(dim=(1, 2)),                          # cluster tendency
        ]
        return torch.stack(feats, dim=1)               # (B,7)

    # --- loss -----------------------------------------------------------------
    def forward(self, fake, real):
        """fake, real: (B,1,H,W) generator output and target, any range."""
        losses = []
        for f_map, r_map in zip(self._hessian_maps(fake), self._hessian_maps(real)):
            # quantize both with the reference range so levels mean the same thing
            lo = r_map.amin(dim=(1, 2, 3), keepdim=True).detach()
            hi = r_map.amax(dim=(1, 2, 3), keepdim=True).detach()
            scale = (hi - lo).clamp_min(self.eps)
            f_n = ((f_map - lo) / scale).clamp(0, 1)
            r_n = ((r_map - lo) / scale).clamp(0, 1)

            f_feat = self._descriptors(self._glcm(self._memberships(f_n)))
            r_feat = self._descriptors(self._glcm(self._memberships(r_n))).detach()

            # relative difference keeps all 7 descriptors on a comparable scale
            rel = (f_feat - r_feat).abs() / (r_feat.abs() + 1e-3)
            losses.append(rel.mean())
        return sum(losses) / len(losses)

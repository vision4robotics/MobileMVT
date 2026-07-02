"""Trapezoid Convolution (TConv) operators.

This module implements the geometry-aware convolution operators used by
MobileMVT / TConvNet. The core idea is to align the convolutional receptive
field with the trapezoidal contour that a rigid vessel exhibits when observed
from an elevated UAV viewpoint, instead of the fixed square grid of a standard
convolution or the unconstrained offsets of a deformable convolution.

class:`TrapezoidConv` -- the base operator. A normalized trapezoidal
sampling template is transformed per window by an affine matrix (anisotropic
scaling, inclination shear, rotation, translation) predicted from the input
feature. The fractional deviation of each transformed sampling point w.r.t.
the integer lattice is encoded into a per-location modulation map that
re-weights the input feature before the main convolution.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrapezoidConv(nn.Module):
    """Trapezoid Convolution operator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        window_size (int): Side length ``K`` of the non-overlapping sampling
            window. Supported values are 2, 3, 4 and 5. ``K=3`` is the default
            used in the paper.
        s (int): Stride of the final output convolution.
    """

    def __init__(self, in_channels, out_channels, window_size=2, s=2):
        super().__init__()
        self.K = window_size
        k = window_size
        self.s = s

        # Normalized trapezoidal sampling templates, one per supported window
        # size. Each row is a (u, v) coordinate in the canonical window frame;
        # the upper base is wider than the lower base, approximating the
        # foreshortened deck silhouette of a vessel seen from above.
        self.register_buffer("template_pts2", torch.tensor([
            [-1.0,  1.0], [ 1.0,  1.0],
            [-0.5, -1.0], [ 0.5, -1.0],
        ], dtype=torch.float32))

        self.register_buffer("template_pts3", torch.tensor([
            [-1.0,  1.0], [ 0.0,  1.0], [ 1.0,  1.0],
            [-0.75, 0.0], [ 0.0,  0.0], [ 0.75, 0.0],
            [-0.5, -1.0], [ 0.0, -1.0], [ 0.5, -1.0],
        ], dtype=torch.float32))

        self.register_buffer("template_pts4", torch.tensor([
            [-1.0,  1.0], [-1.0/3,  1.0], [1.0/3,  1.0], [-1.0, 1.0],
            [-5.0/6,  1.0/3], [-5.0/18,  1.0/3], [ 5.0/18, 1.0/3], [5.0/6, 1.0/3],
            [-2.0/3, -1.0/3], [-2.0/9, -1.0/3], [2.0/9, -1.0/3], [2.0/3, -1.0/3],
            [-1.0/2, -1.0], [-1.0/6, -1.0], [1.0/6, -1.0], [1.0/2, -1.0],
        ], dtype=torch.float32))

        self.register_buffer("template_pts5", torch.tensor([
            [-1.0,  1.0], [-1.0/2, 1.0], [0.0,  1.0], [1.0/2, 1.0], [-1.0, 1.0],
            [-7.0/8, -1.0/2], [-7.0/16, -1.0/2], [0.0, -1.0/2], [7.0/16, -1.0/2], [7.0/8, -1.0/2],
            [-3.0/4,  0.0], [-3.0/8,  0.0], [ 0.0,  0.0], [3.0/8, 0.0], [3.0/4, 0.0],
            [-5.0/8, -1.0/2], [-5.0/16, -1.0/2], [0.0, -1.0/2], [5.0/16, -1.0/2], [5.0/8, -1.0/2],
            [-1.0/2, -1.0], [-1.0/4, -1.0], [0.0, -1.0], [1.0/4, -1.0], [1.0/2, -1.0],
        ], dtype=torch.float32))

        # Dynamic affine estimator: predicts the six affine parameters
        # (Sx, Sy, angle, shear, Tx, Ty) for every K x K window.
        self.param_conv = nn.Conv2d(in_channels, 6, kernel_size=k, stride=k, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.act = nn.Hardswish(inplace=True)

        # Offset encoder: maps the 4-dim deviation descriptor of each of the
        # K*K sampling points to a per-sample modulation weight.
        self.offset2mask = nn.Conv2d(4 * k * k, k * k, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(k * k)
        self.relu = nn.ReLU(inplace=True)

        # Main convolution producing the output feature map.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.s, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.K
        device = x.device
        dtype = x.dtype

        # 1. Predict per-window affine parameters: [B, win_h, win_w, 6].
        params = self.act(self.bn1(self.param_conv(x)))
        params = params.permute(0, 2, 3, 1).contiguous()

        Sx = F.softplus(params[..., 0]) + 1e-3   # anisotropic scale x (> 0)
        Sy = F.softplus(params[..., 1]) + 1e-3   # anisotropic scale y (> 0)
        angle = params[..., 2]                   # rotation
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        shear = torch.tanh(params[..., 3]) * 0.5  # bounded inclination shear
        Tx = torch.tanh(params[..., 4]) * 0.5     # bounded translation x
        Ty = torch.tanh(params[..., 5]) * 0.5     # bounded translation y

        # 2. Build the combined affine matrix R @ S @ Shear.
        a11 = Sx * cos
        a12 = cos * Sx * shear + sin * Sy
        a21 = -Sx * sin
        a22 = -sin * Sx * shear + cos * Sy

        # 3. Select the trapezoidal template for the current window size.
        if k == 2:
            pts = self.template_pts2.to(device, dtype)
        elif k == 3:
            pts = self.template_pts3.to(device, dtype)
        elif k == 4:
            pts = self.template_pts4.to(device, dtype)
        elif k == 5:
            pts = self.template_pts5.to(device, dtype)
        else:
            raise ValueError(f"Unsupported TrapezoidConv window_size={k}.")
        x_pts, y_pts = pts[:, 0], pts[:, 1]

        a11 = a11.unsqueeze(-1)
        a12 = a12.unsqueeze(-1)
        a21 = a21.unsqueeze(-1)
        a22 = a22.unsqueeze(-1)
        Tx = Tx.unsqueeze(-1)
        Ty = Ty.unsqueeze(-1)

        # 4. Map the template to trapezoid-adapted coordinates: [B, win_h, win_w, K*K].
        x_trans = a11 * x_pts + a12 * y_pts + Tx
        y_trans = a21 * x_pts + a22 * y_pts + Ty

        # 5. Encode the fractional deviation of each sampling point w.r.t. the
        #    integer lattice as a 4-dim descriptor [r_u, r_v, 1-r_u, 1-r_v].
        abs_x_1 = torch.abs(x_trans)
        abs_y_1 = torch.abs(y_trans)
        abs_x_2 = 1 - abs_x_1
        abs_y_2 = 1 - abs_y_1
        offsets = torch.cat([abs_x_1, abs_y_1, abs_x_2, abs_y_2], dim=-1)

        # 6. Encode deviations into a per-window modulation and expand to full
        #    resolution via pixel shuffle.
        mask_in = offsets.permute(0, 3, 1, 2).contiguous()      # [B, 4*K*K, win_h, win_w]
        mask = self.act(self.bn2(self.offset2mask(mask_in)))    # [B, K*K, win_h, win_w]
        mask = F.pixel_shuffle(mask, upscale_factor=k)          # [B, 1, H, W]
        if (mask.shape[2] != H) or (mask.shape[3] != W):
            mask = F.interpolate(mask, size=(H, W), mode='nearest')

        # 7. Modulate the input feature and apply the main convolution.
        x = x * mask
        out = self.act(self.silu(self.conv(x)))
        return out

from math import ceil

import numpy as np
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device).unsqueeze(0)

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttentionWithScores(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                (residual_conv_kernel, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False
            )

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = (
            *x.shape,
            self.heads,
            self.num_landmarks,
            self.pinv_iterations,
            self.eps
        )

        remainder = n % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        padded_n = x.shape[1]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(b, padded_n, h, -1).transpose(1, 2),
            (q, k, v)
        )

        if exists(mask):
            mask = mask.unsqueeze(1)
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        l = ceil(padded_n / m)

        q_landmarks = q.view(b, h, m, l, -1).sum(dim=3)
        k_landmarks = k.view(b, h, m, l, -1).sum(dim=3)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = mask.view(b, 1, m, l).sum(dim=-1)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            attn1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out = out + self.res_conv(v)

        out = out.transpose(1, 2).contiguous().view(b, padded_n, -1)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            cls_index = -n
            cls_attn = (attn1[:, :, cls_index].unsqueeze(-2) @ attn2_inv) @ attn3
            cls_attn = cls_attn[:, :, 0, cls_index + 1:]
            return out, cls_attn

        return out


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttentionWithScores(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + attn_out
            return x, attn

        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]

        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, input_size=1024, feat_size=1024):
        super().__init__()
        self.input_size = input_size
        self.pos_layer = PPEG(dim=feat_size)
        self._fc1 = nn.Sequential(
            nn.Linear(self.input_size, feat_size),
            nn.ReLU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=feat_size)
        self.layer2 = TransLayer(dim=feat_size)
        self.norm = nn.LayerNorm(feat_size)
        self._fc2 = nn.Linear(feat_size, self.n_classes)

    def forward(self, x, return_attn=True):
        if x.dim() == 2:
            h = x.unsqueeze(0)
        elif x.dim() == 3:
            h = x
        else:
            raise ValueError(
                'TransMIL expects feature input with shape [N, D] or [B, N, D], got {}'.format(tuple(x.shape))
            )

        h = h[:, :, :self.input_size]
        h = self._fc1(h)

        original_length = h.shape[1]
        _H = int(np.ceil(np.sqrt(original_length)))
        _W = int(np.ceil(np.sqrt(original_length)))

        add_length = _H * _W - original_length
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        if return_attn:
            h, attn1 = self.layer1(h, return_attn=True)
        else:
            h = self.layer1(h)

        h = self.pos_layer(h, _H, _W)

        if return_attn:
            h, attn2 = self.layer2(h, return_attn=True)
        else:
            h = self.layer2(h)

        h = self.norm(h)[:, 0]

        logits = self._fc2(h)
        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)

        if return_attn:
            attn_dict = {
                'layer1': attn1,
                'layer2': attn2,
                'layer1_patch': attn1[:, :, :original_length],
                'layer2_patch': attn2[:, :, :original_length],
                'layer1_patch_mean': attn1[:, :, :original_length].mean(dim=1),
                'layer2_patch_mean': attn2[:, :, :original_length].mean(dim=1),
            }
            return logits, Y_hat, Y_prob, attn_dict

        return logits, Y_hat, Y_prob


if __name__ == "__main__":
    data = torch.randn((1, 128, 1024)).cuda()
    model = TransMIL(n_classes=2, input_size=1024, feat_size=1024).cuda()
    model.eval()

    with torch.no_grad():
        logits, Y_hat, Y_prob, attn_dict = model(data, return_attn=True)

    print(logits.shape, Y_hat.shape, Y_prob.shape)
    print(attn_dict['layer2_patch_mean'].shape)
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

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
        super(TransMIL, self).__init__()
        self.input_size = input_size
        self.pos_layer = PPEG(dim=feat_size)
        self._fc1 = nn.Sequential(nn.Linear(self.input_size, feat_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=feat_size)
        self.layer2 = TransLayer(dim=feat_size)
        self.norm = nn.LayerNorm(feat_size)
        self._fc2 = nn.Linear(feat_size, self.n_classes)

    def forward(self, x):
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

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)

        h = self.norm(h)[:, 0]

        logits = self._fc2(h)
        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)

        return logits, Y_hat, Y_prob


if __name__ == "__main__":
    data = torch.randn((1, 128, 1024)).cuda()
    model = TransMIL(n_classes=2, input_size=1024, feat_size=1024).cuda()
    print(model.eval())
    logits, Y_hat, Y_prob = model(data)
    print(logits.shape, Y_hat.shape, Y_prob.shape)
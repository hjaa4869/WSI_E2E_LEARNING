import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_custom import resnet50_baseline


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class ABMIL(nn.Module):
    def __init__(self,input_size=1024,size2=512,n_classes=2):
        super(ABMIL, self).__init__()
        self.size1 = input_size

        self.L = size2
        self.D = 256
        self.K = 1

        self.backbone = resnet50_baseline(pretrained=True)

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.size1, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
            # nn.Sigmoid()
        )
        
        self.bin = nn.Parameter(torch.tensor(1.0))
        self.faker = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, feature=None):
        B1, N1, C1, H1, W1 = x.shape
        x = x.view(B1 * N1, C1, H1, W1)
        x = self.backbone(x)
        D1 = x.shape[-1]
        x = x.view(B1, N1, D1)
        x = x[:,:,:self.size1]
        # feature = feature.detach()
        # H = self.bin * x + self.faker * feature
        H = x
        H = self.feature_extractor_part2(H)  # NxL
        # H=x

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = A.transpose(1, 2)   # KxN
        A = F.softmax(A, dim=-1)  # softmax over N

        M = torch.bmm(A, H)  # KxL
        M = M.squeeze(1)
        
        # return M

        logits = self.classifier(M) #B,2
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        Y_hat = torch.argmax(logits, dim=-1)#(B,)
        Y_prob = F.softmax(logits, dim=-1)#(B,2)

        return logits, Y_hat, Y_prob
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict

        # return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


# import pdb
# from models.resnet_custom import resnet50_baseline
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MaxPooling(nn.Module):
#     def __init__(self, n_classes, input_size=1024,feat_size=1024):
#         super(MaxPooling, self).__init__()
#         self.input_size = input_size
#         self._fc1 = nn.Sequential(nn.Linear(input_size,feat_size), nn.ReLU())
#         self.n_classes = n_classes
#         self._fc0 = nn.Linear(feat_size, 1)
#         self._fc2 = nn.Linear(feat_size, self.n_classes)
#         self.backbone = resnet50_baseline(pretrained=True)
        
#     def forward(self, x):
        
#         x = self.backbone(x)
#         h = x[:, :self.input_size].unsqueeze(0)

#         h = self._fc1(h) 
#         h = torch.max(h, dim=1)[0]



#         logits = self._fc2(h) #[B, n_classes]
#         Y_hat = torch.argmax(logits, dim=-1)
#         Y_prob = F.softmax(logits, dim=-1)
#         return logits, Y_hat, Y_prob

# import pdb

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from nystrom_attention import NystromAttention


# class TransLayer(nn.Module):

#     def __init__(self, norm_layer=nn.LayerNorm, dim=512):
#         super().__init__()
#         self.norm = norm_layer(dim)
#         self.attn = NystromAttention(
#             dim = dim,
#             dim_head = dim//8,
#             heads = 8,
#             num_landmarks = dim//2,    # number of landmarks
#             pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
#             residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
#             dropout=0.1
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm(x))

#         return x


# class PPEG(nn.Module):
#     def __init__(self, dim=512):
#         super(PPEG, self).__init__()
#         self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
#         self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
#         self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

#     def forward(self, x, H, W):
#         B, _, C = x.shape
#         cls_token, feat_token = x[:, 0], x[:, 1:]
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
#         x = x.flatten(2).transpose(1, 2)
#         x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
#         return x


# class TransMIL(nn.Module):
#     def __init__(self, n_classes, input_size=1024,feat_size=1024):
#         super(TransMIL, self).__init__()
#         self.input_size = input_size
#         self.pos_layer = PPEG(dim=feat_size)
#         self._fc1 = nn.Sequential(nn.Linear(self.input_size,feat_size), nn.ReLU())
#         self.cls_token = nn.Parameter(torch.randn(1, 1, feat_size))
#         self.n_classes = n_classes
#         self.layer1 = TransLayer(dim=feat_size)
#         self.layer2 = TransLayer(dim=feat_size)
#         self.norm = nn.LayerNorm(feat_size)
#         self._fc2 = nn.Linear(feat_size, self.n_classes)
#         self.backbone = resnet50_baseline(pretrained=True)


#     def forward(self, x):
#         # pdb.set_trace()
#         x = self.backbone(x)
#         h = x[:, :self.input_size].unsqueeze(0)
#         h = self._fc1(h) #[B, n, 512]
#         # import pdb;pdb.set_trace()

#         H = h.shape[1]
#         _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
#         add_length = _H * _W - H
#         h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

#         #---->cls_token
#         B = h.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
#         # import pdb;pdb.set_trace()
#         h = torch.cat((cls_tokens, h), dim=1)

#         #---->Translayer x1
#         h = self.layer1(h) #[B, N, 512]

#         #---->PPEG
#         h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
#         #---->Translayer x2
#         h = self.layer2(h) #[B, N, 512]
#         #---->cls_token
#         h = self.norm(h)[:,0]

#         #---->predict
#         logits = self._fc2(h)  # [B, n_classes]
#         # pdb.set_trace()
#         Y_hat = torch.argmax(logits, dim=-1)
#         Y_prob = F.softmax(logits, dim=-1)
#         # Y_hat = torch.topk(logits, 1, dim=1)[1]
#         return logits, Y_hat, Y_prob

# if __name__ == "__main__":
#     data = torch.randn((1, 128, 512)).cuda()
#     model = TransMIL(n_classes=2).cuda()
#     print(model.eval())
#     results_dict = model(data = data)
#     pdb.set_trace()
#     print(results_dict)

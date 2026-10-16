import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 512), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 512)

        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        self.fcc = nn.Linear(input_size, output_class)

    def forward(self, feats, c):  # feats: N x K, c: N x C
        device = feats.device
        feats = self.lin(feats)

        V = self.v(feats).view(feats.shape[0], -1)  # N x V
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q

        _, m_indices = torch.sort(c, dim=0, descending=True)  # N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # C x K

        q_max = self.q(m_feats).reshape(1, -1)  # 1 x (C*Q)

        A = torch.mm(Q, q_max.transpose(0, 1))  # N x C
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
            0
        )

        B = torch.mm(A.transpose(0, 1), V)  # C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x output_class
        C = C.view(1, -1)

        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, input_size=1024, n_classes=2):
        super(DSMIL, self).__init__()
        self.input_size = input_size

        i_classifier = FCLayer(self.input_size, 1)
        b_classifier = BClassifier(input_size=self.input_size, output_class=n_classes)

        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        if x.dim() == 3:
            if x.size(0) != 1:
                raise ValueError('DSMIL expects batch size 1 for bag input, got {}'.format(x.size(0)))
            x = x.squeeze(0)

        x = x[:, :self.input_size]

        feats, classes = self.i_classifier(x)
        logits, A, B = self.b_classifier(feats, classes)

        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)

        return logits, Y_hat, Y_prob


if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = DSMIL(input_size=1024, n_classes=2).cuda()
    print(model.eval())
    logits, Y_hat, Y_prob = model(data)
    print(logits.shape, Y_hat.shape, Y_prob.shape)
import torch
import torch.nn as nn
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)

        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, shape, nhid=512, ncond=0):
        super(Encoder, self).__init__()
        c, h, w = shape
        self.encode = nn.Sequential(
            nn.Conv2d(c, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SelfAttention(128, 13),

            nn.Conv2d(128, 256, 5, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SelfAttention(256, 5),

            nn.Conv2d(256, 512, 5, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Flatten(start_dim=1),
        )
        flat_dimension = 512  # TODO hardcoded for now...
        self.calc_mean = MLP([flat_dimension + ncond, nhid], last_activation=False)  # nhid
        self.calc_logvar = MLP([flat_dimension + ncond, nhid], last_activation=False)  # nhid

    def forward(self, x, y=None):
        x = self.encode(x)
        if y is None:
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))


class Decoder(nn.Module):
    def __init__(self, shape, nhid=512, ncond=0):
        super(Decoder, self).__init__()
        c, h, w = shape
        h2, h4, h8, h16 = int(h / 2), int(h / 4), int(h / 8), int(h / 16)

        self.latent_size = nhid + ncond
        self.main = nn.Sequential(
            nn.Linear(self.latent_size, 256 * h8 * h8),
            nn.Unflatten(1, (256, h8, h8)),  # 256 x h/8 x h/8
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 256, 2, 2),  # 256 x h/4 x h/4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SelfAttention(256, 16),

            nn.ConvTranspose2d(256, 128, 2, 2),  # 128 x h/2 x h/2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SelfAttention(128, 32),

            nn.ConvTranspose2d(128, 32, 2, 2),  # 32 x h x h,
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, c, 5, 1, 2),  # 3 x h x h
            nn.Sigmoid()
        )

    def forward(self, z, y=None):
        if y is not None:
            z = torch.cat((z, y), dim=1)

        # z = z.view(-1, self.latent_size, 1, 1)
        return self.main(z)


class Attention_cVAE(nn.Module):
    def __init__(self, shape, nclass, nhid=16, ncond=16):
        super(Attention_cVAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid, ncond=ncond)
        self.label_embedding = nn.Embedding(nclass, ncond)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y):
        y = self.label_embedding(y)
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar

    def generate(self, y):
        if (type(y) is int):
            y = torch.tensor(y)
        y = y.to(device)
        if (len(y.shape) == 0):
            batch_size = None
            y = y.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = y.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device)
            y = self.label_embedding(y)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res


BCE_loss = nn.BCELoss(reduction="sum")


def loss(X, X_hat, mean, logvar):
    reconstruction_loss = BCE_loss(X_hat, X)
    KL_divergence = torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    return reconstruction_loss + 0.5 * KL_divergence

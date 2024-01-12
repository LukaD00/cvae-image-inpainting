import torch
import torch.nn as nn
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


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
    def __init__(self, shape, nhid=16, ncond=0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w - 8) // 2 - 4) // 2
        hh = ((h - 8) // 2 - 4) // 2
        self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding=0), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                    nn.Conv2d(16, 32, 5, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2),
                                    Flatten(), MLP([ww * hh * 64, 256, 128])
                                    )
        self.calc_mean = MLP([128 + ncond, 64, nhid], last_activation=False)
        self.calc_logvar = MLP([128 + ncond, 64, nhid], last_activation=False)

    def forward(self, x, y=None):
        x = self.encode(x)
        if y is None:
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))


class Decoder(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0):
        super(Decoder, self).__init__()
        self.shape = shape

        self.latent_size = nhid + ncond
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, y=None):
        if y is not None:
            z = torch.cat((z, y), dim=1)

        z = z.view(-1, self.latent_size, 1, 1)
        return self.main(z)


class cVAE(nn.Module):
    def __init__(self, shape, nclass, nhid=16, ncond=16):
        super(cVAE, self).__init__()
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

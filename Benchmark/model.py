import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, hidden_size=768, bottleneck_size=64):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, bottleneck_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.criterion(x_hat, x)
        return loss


class VAE(nn.Module):
    def __init__(self, hidden_size=768, bottleneck_size=64):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(hidden_size // 4, bottleneck_size)
        # hidden => logvar
        self.fc2 = nn.Linear(hidden_size // 4, bottleneck_size)

        self.criterion = nn.MSELoss()

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def resample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu  # use mean to represent the latent variable during inference

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.resample(mu, logvar)
        x_hat = self.decode(z)
        loss = self.loss(x, x_hat, mu, logvar)
        return loss

    def loss(self, x, x_hat, mu, logvar, beta=1):
        recon_loss = self.criterion(x_hat, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_div

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.spectral_norm import spectral_norm


class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            return loss_real + loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = nn.Linear(128, 1, bias=False)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        spectral_norm(self.linear)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


if __name__ == '__main__':
    net_D = ResDiscriminator32().to('cuda')
    loss_fn = Hinge()
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)
    sched_D = optim.lr_scheduler.LambdaLR(
        optim_D, lambda step: 1 - step / FLAGS.total_steps)
    for _ in range(FLAGS.n_dis):
        with torch.no_grad():
            z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
            fake = net_G(z).detach()
        real = next(looper).to(device)
        net_D_real = net_D(real)
        net_D_fake = net_D(fake)
        loss = loss_fn(net_D_real, net_D_fake)

        optim_D.zero_grad()
        loss.backward()
        optim_D.step()


    loss = loss_fn(net_D(net_G(z)))
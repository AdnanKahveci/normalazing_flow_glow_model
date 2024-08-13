import torch
import torch.nn as nn
import numpy as np

# Cihaz ayarı: CUDA mevcutsa GPU kullan, aksi takdirde CPU kullan
device = "cuda" if torch.cuda.is_available() else "cpu"

class ActNorm(nn.Module):
    def __init__(self, channels, device):
        super(ActNorm, self).__init__()
        size = (1, channels, 1, 1)
        self.logs = torch.nn.Parameter(torch.zeros(size, dtype=torch.float, device=device, requires_grad=True))
        self.b = torch.nn.Parameter(torch.zeros(size, dtype=torch.float, device=device, requires_grad=True))
        self.initialized = False
    
    def initialize(self, x):
        if not self.training:
            return
        with torch.no_grad():
            b_ = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            s_ = ((x - b_)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            self.b.data.copy_(-b_)
            self.logs.data.copy_(-torch.log(torch.sqrt(s_)))
            self.initialized = True

    def apply_bias(self, x, reverse):
        if reverse:
            return x - self.b
        return x + self.b
    
    def apply_scale(self, x, logdet, reverse):
        if reverse:
            x = x * torch.exp(-self.logs)
        else:
            x = x * torch.exp(self.logs)
            n, c, h, w = x.size()
            logdet += h * w * self.logs.sum()
        return x, logdet
    
    def forward(self, x, logdet=None, reverse=False):
        if not self.initialized:
            self.initialize(x)
        x = self.apply_bias(x, reverse)
        x, logdet = self.apply_scale(x, logdet, reverse)
        if not reverse:
            loss_mean = x.mean(dim=(0, 2, 3)).mean()
            loss_std = ((x - loss_mean)**2).mean(dim=(0, 2, 3)).mean()
            actnormloss = torch.abs(loss_mean) + torch.abs(1 - loss_std)
            return x, logdet, actnormloss
        return x

if __name__ == "__main__":
    ActNorm(3, device)

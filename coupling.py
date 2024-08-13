import torch
import torch.nn as nn
import numpy as np
from net import NN

# Cihaz ayarı: CUDA mevcutsa GPU kullan, aksi takdirde CPU kullan
device = "cuda" if torch.cuda.is_available() else "cpu"

class CouplingLayer(nn.Module):
    def __init__(self, channels, coupling, device, nn_init_last_zeros=False):
        super(CouplingLayer, self).__init__()
        self.coupling = coupling
        self.channels = channels
        
        if self.coupling == "affine":
            self.net = NN(channels_in=self.channels//2, channels_out=self.channels,
                          device=device, init_last_zeros=nn_init_last_zeros)
        elif self.coupling == "additive":
            self.net = NN(channels_in=self.channels//2, channels_out=self.channels//2,
                          device=device, init_last_zeros=nn_init_last_zeros)
        else:
            raise ValueError("Only 'affine' and 'additive' coupling are implemented.")
        self.to(device)
        
    def forward(self, x, logdet=None, reverse=False):
        n, c, h, w = x.size()
        
        if self.coupling == "affine":
            xa, xb = self.split(x, "split-by-chunk")
            s_and_t = self.net(xb)
            s, t = self.split(s_and_t, "split-by-alternating")
            s = torch.sigmoid(s + 2.)
            
            if not reverse:
                ya = s * xa + t
                y = torch.cat([ya, xb], dim=1)
                logdet = logdet + torch.log(s).view(n, -1).sum(-1) if logdet is not None else None
            else:
                ya = (xa - t) / s
                y = torch.cat([ya, xb], dim=1)
            
        elif self.coupling == "additive":
            xa, xb = self.split(x, "split-by-chunk")
            t = self.net(xb)
            
            if not reverse:
                ya = xa + t
                y = torch.cat([ya, xb], dim=1)
            else:
                ya = xa - t
                y = torch.cat([ya, xb], dim=1)
        
        return y, logdet

    def split(self, x, mode):
        if mode == "split-by-chunk":
            return x[:, :self.channels//2, :, :], x[:, self.channels//2:, :, :]
        elif mode == "split-by-alternating":
            return x[:, 0::2, :, :].contiguous(), x[:, 1::2, :, :].contiguous()

if __name__ == "__main__":
    CouplingLayer(3, "affine", device)

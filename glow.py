import torch
import torch.nn as nn
from flow import Flow
from squeeze import Squeeze
from split import Split
import numpy as np

# Cihaz ayarı: CUDA mevcutsa GPU kullan, aksi takdirde CPU kullan
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Glow(nn.Module):
    
    def __init__(self, image_shape, K, L, coupling, device, n_bits_x=8, nn_init_last_zeros=False):
        super(Glow, self).__init__()
        self.image_shape = image_shape
        self.K = K  # Akış sayısı
        self.L = L  # Katman sayısı
        self.coupling = coupling  # Coupling türü
        self.n_bits_x = n_bits_x  # Giriş görüntüsü için bit sayısı
        self.device = device  # Cihaz
        self.init_resizer = False  # Resizer'ın başlatılıp başlatılmadığını kontrol eder
        self.nn_init_last_zeros = nn_init_last_zeros  # Son katman ağırlıklarını sıfırla
        
        # Katmanları kurma
        c, w, h = image_shape
        self.glow_modules = nn.ModuleList()
        
        for l in range(L - 1):
            # Akış adımları
            squeeze = Squeeze(factor=2)  # Boyut sıkıştırma
            c = c * 4  # Kanal sayısını artırma
            self.glow_modules.append(squeeze)
            for k in range(K):
                flow = Flow(c, self.coupling, device, nn_init_last_zeros)                
                self.glow_modules.append(flow)
            split = Split()  # Ayrıştırma
            c = c // 2  # Kanal sayısını azaltma
            self.glow_modules.append(split)
        # L-th akışı
        squeeze = Squeeze(factor=2)
        c = c * 4
        self.glow_modules.append(squeeze)
        flow = Flow(c, self.coupling, device, nn_init_last_zeros)
        self.glow_modules.append(flow)
        
        # Son işlem
        self.to(device)
    
    def forward(self, x, logdet=None, reverse=False, reverse_clone=True):
        if not reverse:
            n, c, h, w = x.size()
            Z = []
            if logdet is None:
                logdet = torch.tensor(0.0, requires_grad=False, device=self.device, dtype=torch.float)
            for i in range(len(self.glow_modules)):
                module_name = self.glow_modules[i].__class__.__name__
                if module_name == "Squeeze":
                    x, logdet = self.glow_modules[i](x, logdet=logdet, reverse=False)
                elif module_name == "Flow":
                    x, logdet, actloss = self.glow_modules[i](x, logdet=logdet, reverse=False)
                elif module_name == "Split":
                    x, z = self.glow_modules[i](x, reverse=False)
                    Z.append(z)
                else:
                    raise Exception("Bilinmeyen Katman")
            Z.append(x)
            
            if not self.init_resizer:
                self.sizes = [t.size() for t in Z]
                self.init_resizer = True
            return Z, logdet, actloss
        
        if reverse:
            if reverse_clone:
                x = [x[i].clone().detach() for i in range(len(x))]
            else:
                x = [x[i] for i in range(len(x))]
            x_rev = x[-1]  # Burada x z -> latent vektör
            k = len(x) - 2
            for i in range(len(self.glow_modules) - 1, -1, -1):
                module_name = self.glow_modules[i].__class__.__name__
                if module_name == "Split":
                    x_rev = self.glow_modules[i](x_rev, x[k], reverse=True)
                    k = k - 1
                elif module_name == "Flow":
                    x_rev = self.glow_modules[i](x_rev, reverse=True)
                elif module_name == "Squeeze":
                    x_rev = self.glow_modules[i](x_rev, reverse=True)
                else:
                    raise Exception("Bilinmeyen Katman")
            return x_rev
        
    def nll_loss(self, x, logdet=None):
        n, c, h, w = x.size()
        z, logdet, actloss = self.forward(x, logdet=logdet, reverse=False)
        if not self.init_resizer:
            self.sizes = [t.size() for t in z]
            self.init_resizer = True
        z_ = [z_.view(n, -1) for z_ in z]
        z_ = torch.cat(z_, dim=1)
        mean = 0
        logs = 0
        logdet += float(-np.log(256.) * h * w * c)
        logpz = -0.5 * (logs * 2. + ((z_ - mean) ** 2) / np.exp(logs * 2.) + float(np.log(2 * np.pi))).sum(-1)
        nll = -(logdet + logpz).mean()
        nll = nll / float(np.log(2.) * h * w * c)
        return nll, -logdet.mean().item(), -logpz.mean().item(), z_.mean().item(), z_.std().item()
    
    def preprocess(self, x, clone=False):
        if clone:
            x = x.detach().clone()
        n_bins = 2 ** self.n_bits_x
        x = torch.floor(x / 2 ** (8 - self.n_bits_x))
        x = x / n_bins - .5
        x = x + torch.tensor(np.random.uniform(0, 1 / n_bins, x.size()), dtype=torch.float, device=self.device)
        return x
    
    def postprocess(self, x, floor_clamp=True):
        n_bins = 2 ** self.n_bits_x
        if floor_clamp:
            x = torch.floor((x + 0.5) * n_bins) * (1. / n_bins)
            x = torch.clamp(x, 0, 1)
        else:
            x = x + 0.5
        return x
    
    def generate_z(self, n, mu=0, std=1, to_torch=True):
        # z'yi yeniden şekillendirerek geri döndürme yöntemine uygun hale getirme
        z_np = [np.random.normal(mu, std, [n] + list(size)[1:]) for size in self.sizes]
        if to_torch:
            z_t = [torch.tensor(t, dtype=torch.float, device=self.device, requires_grad=False) for t in z_np]
            return z_np, z_t
        else:
            return z_np
        
    def flatten_z(self, z):
        n = z[0].size()[0]
        z_ = [z_.view(n, -1) for z_ in z]
        z_ = torch.cat(z_, dim=1)
        return z_
        
    def unflatten_z(self, z, clone=True):
        # z tensor olmalı
        n_elements = [np.prod(s[1:]) for s in self.sizes]
        z_unflatten = []
        start = 0
        for n, size in zip(n_elements, self.sizes):
            end = start + n
            z_ = z[:, start:end].view([-1] + list(size)[1:])
            if clone:
                z_ = z_.clone().detach()
            z_unflatten.append(z_)
            start = end
        return z_unflatten
    
    def set_actnorm_init(self):
        # Actnorm'ı başlatma
        for i in range(len(self.glow_modules)):
            module_name = self.glow_modules[i].__class__.__name__
            if module_name == "Flow":
                self.glow_modules[i].actnorm.initialized = True

if __name__ == "__main__":
    Glow((3, 32, 32), 4, 4, "affine", device)

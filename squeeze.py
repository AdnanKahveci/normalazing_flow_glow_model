import torch
import torch.nn as nn
import numpy as np

# Cihaz ayarı: CUDA mevcutsa GPU kullan, aksi takdirde CPU kullan
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
class Squeeze(nn.Module):
    
    def __init__(self, factor):
        super(Squeeze, self).__init__()
        self.factor = factor  # Sıkıştırma faktörü
    
    def forward(self, x, logdet=None, reverse=False):
        n, c, h, w = x.size()
        
        if not reverse:
            if self.factor == 1:
                return x, logdet
            # Sıkıştırma işlemi tek satırda yapılır, orijinal kodun aksine
            assert h % self.factor == 0 and w % self.factor == 0, "h,w faktöre bölünemez: h=%d, faktör=%d" % (h, self.factor)
            x = x.view(n, c * self.factor * self.factor, h // self.factor, w // self.factor)
            return x, logdet
        
        if reverse:
            if self.factor == 1:
                return x
            assert c % self.factor**2 == 0, "kanallar faktörün karesine bölünemez"
            # Geri açma işlemi de tek satırda yapılır, orijinal kodun aksine
            x = x.view(n, c // (self.factor**2), h * self.factor, w * self.factor)
            return x
    
if __name__ == "__main__":
    Squeeze(2)

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
    
    
class InvertibleConvolution(nn.Module):
    
    
    def __init__(self, channels, device, ):
        super(InvertibleConvolution, self).__init__()
        
        self.W_size = [channels, channels, 1, 1]
        W_init      = np.random.normal(0,1,self.W_size[:-2])
        W_init      = np.linalg.qr(W_init)[0].astype(np.float32)
        W_init      = W_init.reshape(self.W_size)
        self.W      = nn.Parameter(torch.tensor(W_init, dtype=torch.float,device=device,requires_grad=True))
        
        self.to(device)
    
    
    def forward(self, x, logdet=None, reverse=False):
        n,c,h,w = x.size()
        if not reverse:
            x = F.conv2d(x,self.W)
            detW = torch.slogdet(self.W.squeeze())[1]
            logdet = logdet + h*w*detW
            return x, logdet
        
        if reverse:
            inv_w = torch.inverse(self.W.squeeze().double()).float().view(self.W_size)
            x = F.conv2d(x, inv_w)
            assert not np.isinf(x.mean().item()), "inf after 1x1 conv in reverse"
            assert not np.isnan(x.mean().item()), "nan after 1x1 conv in reverse"
            return x
        

if __name__ == "__main__":
    InvertibleConvolution(3,device)
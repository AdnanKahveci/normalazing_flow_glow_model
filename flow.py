import torch
import torch.nn as nn
import numpy as np
from actnorm import ActNorm
from invertibe_conv import InvertibleConvolution
from coupling import CouplingLayer


# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Flow(nn.Module):
    
    def __init__(self,channels, coupling, device, nn_init_last_zeros=False):
        super(Flow, self).__init__()
        self.actnorm  = ActNorm(channels,device)
        self.coupling = CouplingLayer(channels, coupling, device, nn_init_last_zeros)
        self.invconv  = InvertibleConvolution(channels, device)
        self.to(device)        

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            x, logdet, actnormloss = self.actnorm(x, logdet=logdet, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after actnorm in forward"
            assert not np.isinf(x.mean().item()), "inf after actnorm in forward"
            assert not np.isnan(logdet.sum().item()), "nan in log after actnorm in forward"
            assert not np.isinf(logdet.sum().item()), "inf in log after actnorm in forward"
            
            x, logdet = self.invconv(x, logdet=logdet, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after invconv in forward"
            assert not np.isinf(x.mean().item()), "inf after invconv in forward"
            assert not np.isnan(logdet.sum().item()), "nan in log after invconv"
            assert not np.isinf(logdet.sum().item()), "inf in log after invconv"
            
            x, logdet = self.coupling(x, logdet=logdet, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after coupling in forward"
            assert not np.isinf(x.mean().item()), "inf after coupling in forward"
            assert not np.isnan(logdet.sum().item()), "nan in log after coupling"
            assert not np.isinf(logdet.sum().item()), "inf in log after coupling"

            return x, logdet, actnormloss
        
        if reverse:
            x = self.coupling(x, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after coupling in reverse"
            assert not np.isinf(x.mean().item()), "inf after coupling in reverse"
            
            x = self.invconv(x,  reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after invconv in reverse"
            assert not np.isinf(x.mean().item()), "inf after invconv in reverse"
                        
            x = self.actnorm(x,  reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after actnorm in reverse"
            assert not np.isinf(x.mean().item()), "inf after actnorm in reverse"
            return x

    
if __name__ == "__main__":
    Flow(3,"affine",device)
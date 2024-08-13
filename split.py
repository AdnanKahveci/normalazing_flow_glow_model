import numpy as np
import torch
import torch.nn as nn


# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()
    
    def forward(self, x, y = None, reverse=False):
        n,c,h,w = x.size()
        if not reverse:
            x1 = x[:,:c//2,:,:]
            x2 = x[:,c//2:,:,:]
            return x1, x2
        if reverse:
            assert y is not None, "y must be given"
            x = torch.cat([x, y], dim=1)
            return x
    
if __name__ == "__main__":
    Split()
    
    
    
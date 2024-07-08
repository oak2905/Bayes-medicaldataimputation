import torch
from torch import nn
import math

class BayesianModule(nn.Module):
    """
    creates base class for BNN, in order to enable specific behavior
    """
    def init(self):
        super().__init__()

class BayesianRNN(BayesianModule):
    """
    implements base class for B-RNN to enable posterior sharpening
    """
    def __init__(self,
                 sharpen=False):
        super().__init__()
        
        self.weight_ih_mu = None
        self.weight_hh_mu = None
        self.bias = None
        
        self.weight_ih_sampler = None
        self.weight_hh_sampler = None
        self.bias_sampler = None

        self.weight_ih = None
        self.weight_hh = None
        self.bias = None
        
     
    def sample_weights(self):
        pass
    
    
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class TrainableRandomDistribution(nn.Module):
    #Samples weights for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)
    #Calculates the variational posterior part of the complexity part of the loss
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        # self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi
        #self.normal = torch.distributions.Normal(0, 1)

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is 
        a function from a trainable parameter, and adding a mean

        sets those weights as the current ones

        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        # self.eps_w.data.normal_()
        device = self.mu.device
        epsilon = torch.distributions.Normal(0,1).sample(self.rho.size()).to(device)
        self.sigma = torch.log1p(torch.exp(self.rho)).to(device)
        self.w = self.mu + self.sigma * epsilon
        return self.w

    def log_posterior(self, w=None):

        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """

        assert (self.w is not None), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w
        
        log_sqrt2pi = np.log(np.sqrt(2*self.pi))
        log_posteriors =  -log_sqrt2pi - torch.log(self.sigma) - (((w - self.mu) ** 2)/(2 * self.sigma ** 2)) - 0.5
        return log_posteriors.sum()

class PriorWeightDistribution(nn.Module):
    #Calculates a Scale Mixture Prior distribution for the prior part of the complexity cost on Bayes by Backprop paper
    def __init__(self,
                 pi=0.5,
                 sigma1=1,
                 sigma2=0.002):
        super().__init__()


        
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = torch.distributions.Normal(0, sigma1)
        self.dist2 = torch.distributions.Normal(0, sigma2)

        

        

    def log_prior(self, w):
        """
        Calculates the log_likelihood for each of the weights sampled relative to a prior distribution as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """
        prob_n1 = torch.exp(self.dist1.log_prob(w))

        
        prob_n2 = torch.exp(self.dist2.log_prob(w))
        
        
        # Prior of the mixture distribution, adding 1e-6 prevents numeric problems with log(p) for small p
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6

        return (torch.log(prior_pdf) - 0.5).sum()
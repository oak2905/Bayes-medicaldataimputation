import torch
import numpy as np
import math
from bayesian.utils.weight import TrainableRandomDistribution
from bayesian.loss.kl_divergence import kl_divergence_from_nn
from bayesian.models.bayesian_module import BayesianModule

def variational_estimator(nn_class):
    def nn_kl_divergence(self):
        return kl_divergence_from_nn(self)
    
    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo(self,
                    inputs,
                    labels,
                    criterion,
                    sample_nbr,
                    complexity_cost_weight=1):

        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            loss += criterion(outputs, labels)
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return torch.sqrt(loss*loss / sample_nbr)

    setattr(nn_class, "sample_elbo", sample_elbo)
    return nn_class
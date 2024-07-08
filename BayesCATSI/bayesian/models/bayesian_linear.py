import torch
from torch import nn
from torch.nn import functional as F
from bayesian.models.bayesian_module import BayesianModule
from bayesian.utils.weight import TrainableRandomDistribution, PriorWeightDistribution


class BayesianLinear(BayesianModule):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1 = 1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 0.5,
                 posterior_mu_init = 0,
                 posterior_rho_init = -7.0,
                 bias_mu_init = 0.1,
                 weight_mu_init = 0.1,
                 bias_rho_init = 0.1,
                 weight_rho_init = 0.1):
        super().__init__()

        #our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias_mu_init = bias_mu_init
        self.weight_mu_init = weight_mu_init
        self.bias_rho_init = bias_rho_init
        self.weight_rho_init = weight_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        # Variational weight parameters and sample
        # self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1)) ####
        # self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1)) ####
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(posterior_mu_init-weight_mu_init, posterior_mu_init+weight_mu_init)) ####
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(posterior_rho_init-weight_rho_init,posterior_rho_init+weight_rho_init)) ####
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # Variational bias parameters and sample
        # self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1)) ####
        # self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1)) ####
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(posterior_mu_init-bias_mu_init, posterior_mu_init+bias_mu_init)) ####
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(posterior_rho_init-bias_rho_init, posterior_rho_init+bias_rho_init)) ####
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Sample the weights and forward it

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features), device=x.device)
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior
    
        try:
            # Attempt matrix multiplication
            Y = torch.matmul(x, w.t()) + b
        except RuntimeError as e:
            # Handle the specific error (mat1 and mat2 shapes cannot be multiplied)
            print("Error:", e)
            
            # Perform element-wise multiplication and bias addition as an alternative
            Y = torch.matmul(x, w) + b

        return Y

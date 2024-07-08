import torch
from torch import nn
from torch.nn import functional as F
from bayesian.models.bayesian_module import BayesianModule, BayesianRNN
from bayesian.utils.weight import TrainableRandomDistribution, PriorWeightDistribution


class BayesianLSTMCell(BayesianRNN):
    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 prior_sigma_1 = 1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 0.5,
                 posterior_mu_init = 0,
                 posterior_rho_init = -7.0,
                 peephole = False,
                 bias_mu_init = 0.1,
                 weight_mu_init = 0.1,
                 bias_rho_init = 0.1,
                 weight_rho_init = 0.1,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.peephole = peephole
        
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Variational weight parameters and sample for weight ih
        # self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        # self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).uniform_(posterior_mu_init-weight_mu_init, posterior_mu_init+weight_mu_init))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).uniform_(posterior_rho_init-weight_rho_init,posterior_rho_init+weight_rho_init))
        self.weight_ih_sampler = TrainableRandomDistribution(self.weight_ih_mu, self.weight_ih_rho)
        self.weight_ih = None
        
        # Variational weight parameters and sample for weight hh
        # self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        # self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).uniform_(posterior_mu_init-weight_mu_init, posterior_mu_init+weight_mu_init))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).uniform_(posterior_rho_init-weight_rho_init,posterior_rho_init+weight_rho_init))
        self.weight_hh_sampler = TrainableRandomDistribution(self.weight_hh_mu, self.weight_hh_rho)
        self.weight_hh = None
        
        # Variational weight parameters and sample for bias
        # self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1))
        # self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).uniform_(posterior_mu_init-bias_mu_init, posterior_mu_init+bias_mu_init))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).uniform_(posterior_rho_init-bias_rho_init, posterior_rho_init+bias_rho_init))
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
        self.bias=None
        
        #our prior distributions
        self.weight_ih_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.weight_hh_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
    
        self.log_prior = 0
        self.log_variational_posterior = 0
    
    
    def sample_weights(self):
        #sample weights
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        
        #if use bias, we sample it, otherwise, we are using zeros
        if self.use_bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
            
        else:
            b = None
            b_log_posterior = 0
            b_log_prior = 0
            
        bias = b
        
        #gather weights variational posterior and prior likelihoods
        self.log_variational_posterior = self.weight_hh_sampler.log_posterior() + b_log_posterior + self.weight_ih_sampler.log_posterior()
        
        self.log_prior = self.weight_ih_prior_dist.log_prior(weight_ih) + b_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh)
        
        return weight_ih, weight_hh, bias
 
    def forward(self,
                 x,
                 hidden_states):
       
        weight_ih, weight_hh, bias = self.sample_weights()

        #Assumes x is of shape (batch, feature)
        bs, _ = x.size()
        hidden_seq = []
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            h_t, c_t = (torch.zeros(bs, self.out_features).to(x.device), 
                        torch.zeros(bs, self.out_features).to(x.device))
        else:
            h_t, c_t = hidden_states
        
        #simplifying our out features, and hidden seq list
        HS = self.out_features

        x_t = x
        # batch the computations into a single matrix multiplication
            
        if self.peephole:
                gates = x_t @ weight_ih + c_t @ weight_hh + bias
        else:
                gates = x_t @ weight_ih + h_t @ weight_hh + bias
                g_t = torch.tanh(gates[:, HS*2:HS*3])
            
        i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            
        if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ weight_ih + bias)[:, HS*2:HS*3]
                h_t = torch.tanh(o_t * c_t)
        else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

        return (h_t, c_t)
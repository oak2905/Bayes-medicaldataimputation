import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

##
from bayesian.models import BayesianLinear
from bayesian.loss import variational_estimator

import math
prior_sigma_10 = 1.0
prior_sigma_20 = 0.002
prior_pi0 = 0.5 
bias0=True
posterior_rho_init0 = -7.0
posterior_mu_init0 = 0.0

@variational_estimator
class MLPContext(nn.Module):
    def __init__(self, num_vars, context_hidden):
        super(MLPContext, self).__init__()
        self.num_vars = num_vars
        self.context_hidden = context_hidden
        self.context_mlp = nn.Sequential(
            #nn.Linear(3*self.num_vars+1, 2*context_hidden),
            BayesianLinear(3*self.num_vars+1, 2*context_hidden, prior_sigma_1 = prior_sigma_10, 
            prior_sigma_2=prior_sigma_20, prior_pi =prior_pi0, 
            bias=bias0, posterior_rho_init = posterior_rho_init0,
            posterior_mu_init = posterior_mu_init0, weight_mu_init = 0.1,bias_mu_init=0.1),
            
            nn.ReLU(),
            nn.Linear(2*context_hidden, context_hidden)
        )
    def forward(self, x):
        return self.context_mlp(x)


@variational_estimator
class MLPFeatureImputation(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(MLPFeatureImputation, self).__init__()
        
        self.W = Parameter(torch.Tensor(input_size, hidden_size, input_size))
        self.b = Parameter(torch.Tensor(input_size, hidden_size))
        ##
        self.nonlinear_regression = nn.Sequential(
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            BayesianLinear(hidden_size, hidden_size, prior_sigma_1 = prior_sigma_10, 
                prior_sigma_2=prior_sigma_20, prior_pi =prior_pi0, bias=bias0, posterior_rho_init = posterior_rho_init0,
                posterior_mu_init = posterior_mu_init0, weight_mu_init = 0.1,bias_mu_init=0.1),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
            # BayesianLinear(hidden_size, 1, prior_sigma_1 = 1, prior_sigma_2=0.002, prior_pi = 0.5)
        )

        m = torch.ones(input_size, hidden_size, input_size)
        stdv = 1. / math.sqrt(input_size)
        for i in range(input_size):
            m[i, :, i] = 0
        self.register_buffer('m', m)
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        hidden = torch.cat(tuple(F.linear(x, self.W[i] * Variable(self.m[i]), self.b[i]).unsqueeze(2)
                            for i in range(len(self.W))), dim=2)
        
        z_h = self.nonlinear_regression(hidden)

        return z_h.squeeze(-1)

@variational_estimator
class RNNHistoryImputation(nn.Module):
    def __init__(self, hidden_size, num_vars):
        super(RNNHistoryImputation, self).__init__()
        self.hidden_size = hidden_size
        self.num_vars = num_vars
      

        # self.recurrent_impute = nn.Sequential(
        #                 nn.ReLU(),
        #                 # nn.Linear(2 * self.hidden_size, self.num_vars),
        #                 BayesianLinear(2 * self.hidden_size, self.num_vars, prior_sigma_1 = 1, prior_sigma_2=0.002, prior_pi = 0.5, bias = True),
        #                 nn.ReLU(),
        #                 nn.Linear(self.num_vars, 1)
        #                 # BayesianLinear(self.num_vars, 1, prior_sigma_1 = 1, prior_sigma_2=0.002, prior_pi = 0.5, bias = True)
        # )
        self.recurrent_impute = BayesianLinear(2*self.hidden_size, 
            self.num_vars, prior_sigma_1 = prior_sigma_10, 
            prior_sigma_2=prior_sigma_20, prior_pi =prior_pi0, 
            bias=bias0, posterior_rho_init = posterior_rho_init0,
            posterior_mu_init = posterior_mu_init0, weight_mu_init = 0.1,bias_mu_init=0.1)

    def forward(self, x):
        # hidden = torch.cat(tuple(x.unsqueeze(2) for i in range((self.num_vars))), dim =2)
        # output = self.recurrent_impute(hidden)
       
        # return output.squeeze(-1)
        return self.recurrent_impute(x)

@variational_estimator
class FuseImputation(nn.Module):
    def __init__(self, num_vars):
        super(FuseImputation, self).__init__()
       
        self.num_vars = num_vars
        
        # self.fuse_impute = nn.Linear(2*self.num_vars, self.num_vars)
        self.fuse_impute = BayesianLinear(2*self.num_vars, 
            self.num_vars, prior_sigma_1 = prior_sigma_10, 
            prior_sigma_2=prior_sigma_20, prior_pi =prior_pi0, 
            bias=bias0, posterior_rho_init = posterior_rho_init0,
            posterior_mu_init = posterior_mu_init0, weight_mu_init = 0.1,bias_mu_init=0.1)
    def forward(self, x):
        output = self.fuse_impute(x)
        return output



class InputTemporalDecay(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        return torch.exp(-gamma)

# @variational_estimator
class RNNContext(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.GRUCell(input_size, hidden_size)
        

    def forward(self, input, seq_lengths):
        T_max = input.shape[1]  # batch x time x dims

        h = torch.zeros(input.shape[0], self.hidden_size).to(input.device)
        hn = torch.zeros(input.shape[0], self.hidden_size).to(input.device)
      
        for t in range(T_max):
            # h = self.rnn_cell(input[:, t, :].unsqueeze(dim = 0), h)
            # h = h[0][0]
            h = self.rnn_cell(input[:, t, :], h)
            padding_mask = ((t + 1) <= seq_lengths).float().unsqueeze(1).to(input.device)
            hn = padding_mask * h + (1-padding_mask) * hn

        return hn

## hidden size original = 64 and context hidden = 32

class CATSI(nn.Module):
    def __init__(self, num_vars, hidden_size=64, context_hidden=32):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.context_mlp = MLPContext(self.num_vars, context_hidden)
        self.context_rnn = RNNContext(2*self.num_vars, context_hidden)
        
        self.initial_hidden = nn.Linear(2*context_hidden, 2*hidden_size)
        self.initial_cell_state = nn.Tanh()

        self.rnn_cell_forward = nn.LSTMCell(2*num_vars+2*context_hidden, hidden_size)
        self.rnn_cell_backward = nn.LSTMCell(2*num_vars+2*context_hidden, hidden_size)

        self.decay_inputs = InputTemporalDecay(input_size = num_vars)

        self.recurrent_impute = RNNHistoryImputation(hidden_size = hidden_size, num_vars = num_vars)
        self.feature_impute = MLPFeatureImputation(input_size = num_vars)

        self.fuse_imputations = FuseImputation(num_vars)

    def forward(self, data):
        seq_lengths = data['lengths']

        values = data['values']  # pts x time_stamps x vars
        masks = data['masks']
        deltas = data['deltas']

        # compute context vector, h0 and c0
        T_max = values.shape[1]
        
        padding_masks = torch.cat(tuple(((t + 1) <= seq_lengths).float().unsqueeze(1).to(values.device)
                                   for t in range(T_max)), dim=1)
        padding_masks = padding_masks.unsqueeze(2).repeat(1, 1, values.shape[2])  # pts x time_stamps x vars

        data_means = values.sum(dim=1) / masks.sum(dim=1)  # pts x vars
        data_variance = ((values - data_means.unsqueeze(1)) ** 2).sum(dim=1) / (masks.sum(dim=1) - 1)
        data_stdev = data_variance ** 0.5
        data_missing_rate = 1 - masks.sum(dim=1) / padding_masks.sum(dim=1)
        data_stats = torch.cat((seq_lengths.unsqueeze(1).float(), data_means, data_stdev, data_missing_rate), dim=1)

        ### to avoid divide by zero error
        epsilon = 1e-5

        # normalization
        min_max_norm = data['max_vals'] - data['min_vals'] + epsilon
        normalized_values = (values - data['min_vals']) / min_max_norm
        normalized_means = (data_means - data['min_vals'].squeeze(1)) / min_max_norm.squeeze(1)

        if self.training:
            normalized_evals = (data['evals'] - data['min_vals']) / min_max_norm

        x_prime = torch.zeros_like(normalized_values)
        x_prime[:, 0, :] = normalized_values[:, 0, :]
        for t in range(1, T_max):
            x_prime[:, t, :] = normalized_values[:, t-1, :]

        gamma = self.decay_inputs(deltas)
        x_decay = gamma * x_prime + (1 - gamma) * normalized_means.unsqueeze(1)
        x_complement = (masks * normalized_values + (1-masks) * x_decay) * padding_masks

        context_mlp = self.context_mlp(data_stats)
        context_rnn = self.context_rnn(torch.cat((x_complement, deltas), dim=-1), seq_lengths)
        context_vec = torch.cat((context_mlp, context_rnn), dim=1)
        h = self.initial_hidden(context_vec)
        c = self.initial_cell_state(h)

        inputs = torch.cat([x_complement, masks, context_vec.unsqueeze(1).repeat(1, T_max, 1)], dim=-1)
        
        h_forward, c_forward = h[:, :self.hidden_size], c[:, :self.hidden_size]
        h_backward, c_backward = h[:, self.hidden_size:], c[:, self.hidden_size:]
        hiddens_forward = h[:, :self.hidden_size].unsqueeze(1)
        hiddens_backward = h[:, self.hidden_size:].unsqueeze(1)
        for t in range(T_max-1):
            h_forward, c_forward = self.rnn_cell_forward(inputs[:, t, :],
                                                         (h_forward, c_forward))
            h_backward, c_backward = self.rnn_cell_backward(inputs[:, T_max-1-t, :],
                                                            (h_backward, c_backward))
            hiddens_forward = torch.cat((hiddens_forward, h_forward.unsqueeze(1)), dim=1)
            hiddens_backward = torch.cat((h_backward.unsqueeze(1), hiddens_backward), dim=1)

        
        rnn_imp = self.recurrent_impute(torch.cat((hiddens_forward, hiddens_backward), dim=2))
        feat_imp = self.feature_impute(x_complement).squeeze(-1)

        # imputation fusion
        
        beta = torch.sigmoid(self.fuse_imputations(torch.cat((gamma, masks), dim=-1)))
        imp_fusion = beta * feat_imp + (1 - beta) * rnn_imp
        final_imp = masks * normalized_values + (1-masks) * imp_fusion
        

        ########################################
       

        ## New  Line Added for Bayes version
        criterion_loss = nn.MSELoss(reduction = 'sum')
        sample_nbr = 30
        # rnn_loss = criterion_loss(rnn_imp * masks, normalized_values * masks)
        rnn_loss, outputRNN = self.recurrent_impute.sample_elbo(
                inputs=torch.cat((hiddens_forward, hiddens_backward), dim=2),
                labels=normalized_values * masks,
                criterion=criterion_loss,
                sample_nbr=sample_nbr,  # Number of samples from the posterior distribution
                complexity_cost_weight=1 / rnn_imp.shape[0]  # Weight for complexity cost term
            )
        # feat_loss = criterion_loss(feat_imp * masks, normalized_values * masks)
        feat_loss, outputFeat = self.feature_impute.sample_elbo(
                inputs=feat_imp * masks,
                labels=normalized_values * masks,
                criterion=criterion_loss,
                sample_nbr=sample_nbr,  # Number of samples from the posterior distribution
                complexity_cost_weight=1 / feat_imp.shape[0]  # Weight for complexity cost term
            )
        fusion_loss = criterion_loss(imp_fusion * masks, normalized_values * masks)
  
        total_loss = rnn_loss + feat_loss + fusion_loss
        if not(self.training):
          beta = torch.sigmoid(self.fuse_imputations(torch.cat((gamma, masks), dim=-1)))
          meanValue = beta*outputFeat + (1 - beta)*outputRNN 
          finalLower = torch.quantile(meanValue, 0.05, dim=0)
          finalUpper = torch.quantile(meanValue, 0.95, dim=0)
          finalUpper = masks * normalized_values + (1-masks) * finalUpper
          finalLower = masks * normalized_values + (1-masks) * finalLower
          outputSet = meanValue.permute(1, 2, 3, 0)
          del meanValue

        ##
        if self.training:
            # rnn_loss_eval = criterion_loss(rnn_imp * data['eval_masks'], normalized_evals * data['eval_masks'])
            rnn_loss_eval,_ = self.recurrent_impute.sample_elbo(
                inputs=torch.cat((hiddens_forward, hiddens_backward), dim=2),
                labels=normalized_evals * data['eval_masks'],
                criterion=criterion_loss,
                sample_nbr=sample_nbr,  # Number of samples from the posterior distribution
                complexity_cost_weight=1 / rnn_imp.shape[0]  # Weight for complexity cost term
            )
            # feat_loss_eval = criterion_loss(feat_imp * data['eval_masks'], normalized_evals * data['eval_masks'])
            feat_loss_eval,_ = self.feature_impute.sample_elbo(
                inputs=feat_imp * data['eval_masks'],
                labels=normalized_evals * data['eval_masks'],
                criterion=criterion_loss,
                sample_nbr=sample_nbr,  # Number of samples from the posterior distribution
                complexity_cost_weight=1 / feat_imp.shape[0]  # Weight for complexity cost term
            )
            fusion_loss_eval = criterion_loss(imp_fusion * data['eval_masks'], normalized_evals * data['eval_masks'])
          
            total_loss_eval = rnn_loss_eval + feat_loss_eval + fusion_loss_eval

        def rescale(x):
            return torch.where(padding_masks==1, x * min_max_norm + data['min_vals'], padding_masks)

        feat_imp = rescale(feat_imp)
        rnn_imp = rescale(rnn_imp)
        final_imp = rescale(final_imp)
        if(not(self.training)):
          finalUpper = rescale(finalUpper)
          finalLower = rescale(finalLower)
        out_dict = {
            'loss': total_loss / masks.sum(),
            'verbose_loss': [
                ('rnn_loss', rnn_loss / masks.sum(), masks.sum()),
                ('feat_loss', feat_loss / masks.sum(), masks.sum()),
                ('fusion_loss', fusion_loss / masks.sum(), masks.sum())
            ],
            'loss_count': masks.sum(),
            'imputations': final_imp,
            'feat_imp': feat_imp,
            'hist_imp': rnn_imp 
        }
        if(not(self.training)):
          out_dict['finalUpper'] = finalUpper
          out_dict['finalLower'] = finalLower
          out_dict['outputSet'] = outputSet
        if self.training:
            #print(data['eval_masks'].sum())
            out_dict['loss_eval'] = total_loss_eval / data['eval_masks'].sum()
            out_dict['loss_eval_count'] = data['eval_masks'].sum()
            out_dict['verbose_loss'] += [
                ('rnn_loss_eval', rnn_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum()),
                ('feat_loss_eval', feat_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum()),
                ('fusion_loss_eval', fusion_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum())
            ]

        return out_dict


import torch
import torch.nn as nn

import numpy as np

class RBFLinearCost:
    """
    MMD cost implementation with rff feature representations
    NOTE: Currently hardcoded to cpu
    :param expert_data: (torch Tensor) expert data used for feature matching
    :param feature_dim: (int) feature dimension for rff
    :param input_type: (str) state (s), state-action (sa), state-next state (ss),
                       state-action-next state (sas)
    :param cost_range: (list) inclusive range of costs
    :param bw_quantile: (float) quantile used to fit bandwidth for rff kernel
    :param bw_samples: (int) number of samples used to fit bandwidth
    :param lambda_b: (float) weight parameter for bonus and cost
    :param lr: (float) learning rate for discriminator/cost update. 0.0 = closed form update
    :param seed: (int) random seed to set cost function
    """
    def __init__(self,
                 expert_dataset,
                 cost_cfg,
                 feature_dim=1024,
                 bw_samples=100000):

        # Set random seed
        torch.manual_seed(cost_cfg.seed)
        np.random.seed(cost_cfg.seed)

        self.expert_dataset = expert_dataset
        self.input_type = cost_cfg.input_type
        if self.input_type == 'ss':
            input_dim = 2 * self.expert_dataset.states.size(1)
        else:
            raise NotImplementedError("Only working for (s, s') transitions now.")

        self.feature_dim = cost_cfg.feature_dim
        self.cost_range = cost_cfg.cost_range
        
        if self.cost_range is not None:
            self.c_min, self.c_max = self.cost_range
        
        self.lambda_b = cost_cfg.lambda_b
        self.lr = cost_cfg.lr

        # Fit Bandwidth
        self.quantile = cost_cfg.bw_quantile
        self.bw_samples = bw_samples
        self.bw = self.fit_bandwidth(expert_dataset)

        # Define Phi and Cost weights
        self.rff = nn.Linear(input_dim, feature_dim)
        self.rff.bias.data = (torch.rand_like(self.rff.bias.data)-0.5)*2.0*np.pi
        self.rff.weight.data = torch.rand_like(self.rff.weight.data)/(self.bw+1e-8)

        # W Update Init
        self.w = None

        # Compute Expert Phi Mean
        self.expert_rep = self.get_rep(expert_dataset.states, expert_dataset.next_states)
        self.phi_e = self.expert_rep.mean(dim=0)

    def get_rep(self, states, next_states):
        """
        Returns an RFF representation given an input
        """
        if self.input_type == 'ss':
            x = torch.cat([states, next_states], dim=-1)
        else:
            raise NotImplementedError("Nothing implemented for (s, s') transitions at the moment.")
            
        with torch.no_grad():
            out = self.rff(x.cpu())
            out = torch.cos(out)*np.sqrt(2/self.feature_dim)
        return out

    def fit_bandwidth(self, expert_dataset):
        """
        Uses the median trick to fit the bandwidth for the RFF kernel
        """
        data = torch.cat([expert_dataset.states, expert_dataset.next_states], dim=-1)
        num_data = data.shape[0]
        idxs_0 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        idxs_1 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        norm = torch.norm(data[idxs_0, :]-data[idxs_1, :], dim=1)
        bw = torch.quantile(norm, q=self.quantile).item()
        return bw

    def fit_cost(self, data_pi):
        """
        Updates the weights of the cost with the closed form solution
        """
        phi = self.get_rep(data_pi).mean(0)
        feat_diff = phi - self.phi_e

        # Closed form solution
        self.w = feat_diff

        return torch.dot(self.w, feat_diff).item()

    def get_costs(self, x):
        """
        Returrns the IPM (MMD) cost for a given input
        """
        data = self.get_rep(x)
        if self.cost_range is not None:
            return torch.clamp(torch.mm(data, self.w.unsqueeze(1)), self.c_min, self.c_max)
        return torch.mm(data, self.w.unsqueeze(1))

    def get_expert_cost(self):
        """
        Returns the mean expert cost given our current discriminator weights and representations
        """
        return (1-self.lambda_b)*torch.clamp(torch.mm(self.expert_rep, self.w.unsqueeze(1)), self.c_min, self.c_max).mean()

    def get_bonus_costs(self, states, actions, ensemble, next_states=None):
        """
        Computes the cost with pessimism
        """
        if self.input_type == 'sa':
            rff_input = torch.cat([states, actions], dim=1)
        elif self.input_type == 'ss':
            assert(next_states is not None)
            rff_input = torch.cat([states, next_states], dim=1)
        elif self.input_type == 'sas':
            rff_input = torch.cat([states, actions, next_states], dim=1)
        elif self.input_type == 's':
            rff_input = states
        else:
            raise NotImplementedError("Input type not implemented")

        # Get Linear Cost 
        rff_cost = self.get_costs(rff_input)

        if self.cost_range is not None:
            # Get Bonus from Ensemble
            discrepancy = ensemble.get_action_discrepancy(states, actions)/ensemble.threshold
            discrepancy = discrepancy.view(-1, 1)
            discrepancy[discrepancy>1.0] = 1.0
            # Bonus is LOW if (s,a) is unknown
            bonus = discrepancy * self.c_min
        else:
            bonus = ensemble.get_action_discrepancy(states, actions).view(-1,1)

        # Weight cost components
        ipm = (1-self.lambda_b)*rff_cost

        # Conservative/Pessimism Penalty term
        weighted_bonus = self.lambda_b*bonus.cpu() # Note cpu hardcoding

        # Cost
        cost = ipm - weighted_bonus

        # Logging info
        info = {'bonus': weighted_bonus, 'ipm': ipm, 'v_targ': rff_cost, 'cost': cost}

        return cost, info
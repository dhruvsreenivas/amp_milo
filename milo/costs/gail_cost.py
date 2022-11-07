import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from utils import mlp

class Discriminator(nn.Module):
    '''Discriminator network used for GAIL cost.'''
    def __init__(self, cfg):
        super().__init__()
        self.mlp = mlp(cfg.state_dim, cfg.hidden_sizes, cfg.disc_dim, cfg.activation)
        
        out_dim = cfg.hidden_sizes[-1] if cfg.hidden_sizes else cfg.state_dim
        self.logits_layer = nn.Linear(out_dim, 1)
        
        # weight init is default for mlp (i.e. leave everything alone)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                # zero out bias for init
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
                    
        # logit layer init
        nn.init.uniform_(self.logits_layer.weight, -1.0, 1.0)
        nn.init.zeros_(self.logits_layer.bias)
        
    def forward(self, s):
        output = self.mlp(s)
        return self.logits_layer(output)
    
class GAILCost:
    def __init__(self,
                 expert_dataset,
                 disc_cfg,
                 cost_cfg):
        
        # expert dataset
        self.expert_dataset = expert_dataset
        self.n_expert_samples = len(expert_dataset)
        
        # discriminator
        self.disc = Discriminator(disc_cfg)
        self.disc_opt = optim.Adam(cost_cfg.lr, eps=cost_cfg.eps)
        
        self.scaling_coef = cost_cfg.scaling_coef
        self.reg_coef = cost_cfg.reg_coef
        
        # loss type
        self.loss_type = cost_cfg.loss_type
        if self.loss_type == 'log_likelihood':
            self.disc_loss_fn = nn.BCEWithLogitsLoss()
            self.disc_logit_scale = cost_cfg.disc_logit_scale
        
    def get_disc_weights(self, concat=True):
        weights = []
        for m in self.disc.mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        
        logit_weight = self.disc.logits_layer.weight
        weights.append(torch.flatten(logit_weight))
        
        if concat:
            return torch.cat(weights, dim=-1)
        else:
            return weights
        
    def get_disc_logit_weights(self):
        weight = self.disc.logits_layer.weight
        return torch.flatten(weight)
        
    def sample_from_expert_data(self, batch_size):
        return self.expert_dataset.sample(batch_size)
        
    # losses
    def expert_disc_loss(self, exp_logits):
        if self.loss_type == 'least_squares':
            squared_outs = (1.0 - exp_logits) ** 2
            return torch.mean(squared_outs)
        else:
            labels = torch.ones_like(exp_logits)
            return self.disc_loss_fn(exp_logits, labels)
    
    def agent_disc_loss(self, agent_logits):
        if self.loss_type == 'least_squares':
            squared_outs = (1.0 + agent_logits) ** 2
            return torch.mean(squared_outs)
        else:
            labels = torch.zeros_like(agent_logits)
            return self.disc_loss_fn(agent_logits, labels)
        
    def disc_weight_decay_loss(self):
        weights = self.get_disc_weights()
        return torch.sum(torch.square(weights))
    
    def disc_logit_loss(self):
        logit_weight = self.get_disc_logit_weights()
        return torch.sum(torch.square(logit_weight))
    
    def gradient_penalty(self, expert_obs):
        expert_logits = self.disc(expert_obs)
        expert_grads = grad(expert_logits,
                            inputs=expert_obs,
                            grad_outputs=torch.ones_like(expert_logits),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
        
        expert_grads = torch.sum(torch.square(expert_grads), dim=-1)
        return expert_grads.mean()
    
    def weight_decay_loss(self):
        pass
        
    def disc_accuracy(self, exp_logits, agent_logits):
        agent_acc = agent_logits < 0
        expert_acc = exp_logits > 0
        
        agent_acc = torch.mean(agent_acc.float())
        expert_acc = torch.mean(expert_acc.float())
        
        return agent_acc, expert_acc
    
    @torch.no_grad()
    def get_costs(self, logits):
        if self.loss_type == 'least_squares':
            costs = 1.0 - 0.25 * (logits - 1.0) ** 2
            costs = torch.max(costs, 0.0)
            return costs
        else:
            prob = 1 / (1 + torch.exp(-logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001)))
            disc_r *= self.disc_logit_scale
            return disc_r
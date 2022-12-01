import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from milo.utils import rssm_weight_init

def mlp(input_dim, hidden_sizes, output_dim, activation='relu'):
    '''Standad MLP.'''
    layers = []
    act = nn.ReLU() if activation == 'relu' else nn.Tanh()
    
    dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(dim, size))
        layers.append(act)
        dim = size
    layers.append(nn.Linear(dim, output_dim))
    
    return nn.Sequential(*layers)

class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_update = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, input, state):
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h

class RSSMCell(nn.Module):
    '''Ensemble of RSSM cells that define the model.'''
    def __init__(self,
                 n_models,
                 embed_dim,
                 deter_dim,
                 stoc_dim,
                 stoc_discrete_dim,
                 hidden_dim,
                 action_dim,
                 discrete=False,
                 gru_type='reg',
                 activation='relu'):
        super().__init__()
        
        self.deter_dim = deter_dim
        self.stoc_dim = stoc_dim
        self.stoc_discrete_dim = stoc_discrete_dim
        self.discrete = discrete
        self.feature_dim = self.deter_dim + self.stoc_dim * (self.stoc_discrete_dim if discrete else 1)
        
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Tanh()
        
        # recurrent stack
        self.pre_gru = nn.Linear(self.feature_dim - self.deter_dim + action_dim, hidden_dim)
        if gru_type == 'reg':
            self.gru = nn.GRU(hidden_dim, deter_dim)
        else:
            self.gru = NormGRUCell(hidden_dim, deter_dim)
            
        # decoders
        dist_dim = self.stoc_dim * (self.stoc_discrete_dim if self.discrete else 2)
        prior_mlp_lst = [nn.Sequential(
            nn.Linear(self.deter_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, dist_dim)
        ) for _ in range(n_models)]
        
        self.prior_mlp_lst = nn.ModuleList(prior_mlp_lst)
        
        self.post_mlp = nn.Sequential(
            nn.Linear(self.deter_dim + embed_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, dist_dim)
        )
        
        self.apply(rssm_weight_init)
        
    def init_state(self, batch_size):
        """
        Getting the initial RNN state
        """
        device = next(self.gru.parameters()).device
        if self.discrete:
            state = {
                "deter": torch.zeros((batch_size, self.deter_dim), device=device),
                "stoc": torch.zeros(
                    (batch_size, self.stoc_dim * self.stoc_discrete_dim), device=device
                ),
            }
        else:
            state = {
                "deter": torch.zeros((batch_size, self.deter_dim), device=device),
                "stoc": torch.zeros((batch_size, self.stoc_dim), device=device),
            }
        return state
    
    def prior_forward(self, action, mask, hidden_state, model_idx=None, sample=True):
        k = (
            model_idx if model_idx is not None else np.random.randint(0, self.n_models)
        )  # Same strategy as LOMPO

        deter_state, prev_latent = hidden_state["deter"], hidden_state["stoc"]

        # Reset Masks
        deter_state *= mask
        prev_latent *= mask

        x = torch.cat([prev_latent, action], dim=-1)
        x = self.act(self.pre_gru(x))
        # Note the hidden state encodes deterministic dynamics
        deter_state = self.gru(x, deter_state)
        # Take the k'th model for next imagined latent
        prior = self.prior_mlp[k](deter_state)
        prior_dist, prior_stats = self.zdist(prior)
        if sample:
            sample_latent = prior_dist.rsample().reshape(action.size(0), -1)
        else:
            sample_latent = prior_dist.mean.reshape(action.size(0), -1)
        return prior_stats, {"deter": deter_state, "stoc": sample_latent}

    def zdist(self, post_prior):
        # Either returns One hot Categorical or Multivariate Normal Diagonal
        if self.discrete:
            logits = post_prior.reshape(
                post_prior.shape[:-1] + (self.stoc_dim, self.stoc_discrete_dim)
            )
            stats = {"logits": logits}
        else:
            mean, std = post_prior.chunk(2, -1)
            std = F.softplus(std) + 0.1
            stats = {"mean": mean, "std": std}
        return self.get_dist(stats), stats

    def get_dist(self, stats):
        if self.discrete:
            dist = torch.distributions.OneHotCategoricalStraightThrough(
                logits=stats["logits"]
            )
        else:
            dist = torch.distributions.normal.Normal(stats["mean"], stats["std"])
        return torch.distributions.Independent(dist, 1)

    def get_feature(self, state):
        return torch.cat([state["deter"], state["stoc"]], -1)

    def forward(
        self, embedding, action, mask, hidden_state, model_idx=None, sample=True
    ):
        k = (
            model_idx if model_idx is not None else np.random.randint(0, self.n_models)
        )  # Same strategy as LOMPO

        deter_state, prev_latent = hidden_state["deter"], hidden_state["stoc"]

        # Reset Masks
        deter_state *= mask
        prev_latent *= mask

        x = torch.cat([prev_latent, action], dim=-1)
        x = self.act(self.pre_gru(x))
        # Note the hidden state encodes deterministic dynamics
        deter_state = self.gru(x, deter_state)

        # Prior
        prior = self.prior_mlp[k](deter_state.clone())
        _, prior_stats = self.zdist(prior)

        # Post
        x = torch.cat([deter_state, embedding], dim=-1)
        post = self.post_mlp(x)
        post_dist, post_stats = self.zdist(post)
        if sample:
            post_latent = post_dist.rsample().reshape(action.size(0), -1)
        else:
            post_latent = post_dist.mean.reshape(action.size(0), -1)

        return post_stats, prior_stats, {"deter": deter_state, "stoc": post_latent}
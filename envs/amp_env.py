import isaacgym
from isaacgymenvs.tasks.humanoid_amp import HumanoidAMP
from milo.dynamics_models.ensembles import DynamicsEnsemble

from isaacgymenvs.learning.amp_players import AMPPlayerContinuous
import gym
import torch
import numpy as np
from typing import Dict
from copy import deepcopy

class ModelBasedWrapper(gym.Wrapper):
    def __init__(self,
                 env: HumanoidAMP,
                 cfg: Dict,
                 agent_chkpt: str,
                 ensemble: DynamicsEnsemble,
                 init_state_buffer=None,
                 norm_thresh = float('inf')):
        '''Model based Gym wrapper around VecTasks.'''
        
        super().__init__()
        self.env = env
        self.ensemble = ensemble
        # fully in eval mode
        for model in self.ensemble.models:
            model.eval()
        
        self.train_for_diff = self.ensemble.models[0].train_for_diff
        
        self.horizon = env.max_episode_length
        
        # need to add trained amp agent to get the reward
        params = deepcopy(cfg['params'])
        self.agent = AMPPlayerContinuous(params=params)
        self.agent.restore(agent_chkpt) # load in trained agent
        
        # additional env things
        self.num_steps = 0
        self.reset_counter = 0 # for the model selection
        
        # model based step stuff
        self.model_idx = 0
        self.reset_counter = 0
        
        # init state buffer (TODO do we even need this?)
        self.init_state_buffer = init_state_buffer
        
        # norm threshold so we don't like go into a weird area of the env
        self.norm_thresh = norm_thresh
        
    def reset(self):
        # reset model by choosing the next one in the batch
        self.num_steps = 0
        self.model_idx = (self.model_idx + 1) % self.ensemble.n_models
        
        # reset init state with stuff from buffer every so often
        if self.init_state_buffer is not None:
            idx = np.random.randint(0, len(self.init_state_buffer))
            obs_dict = self.init_state_buffer[idx] if self.reset_counter % 5 == 0 else self.env.reset()
        else:
            obs_dict = self.env.reset()
            
        self.reset_counter += 1
        return obs_dict
    
    def horizon_done(self):
        return self.num_steps >= self.horizon
    
    def norm_done(self):
        state_norms = torch.norm(self.env.obs_buf, p=2, dim=-1) # (num_envs)
        return state_norms >= self.norm_thresh # (num_envs)
    
    def step(self, actions: torch.Tensor):
        obs = self.env.obs_buf # (num_envs, obs_dim)
        
        # next observation
        out = self.ensemble[self.model_idx](obs, actions)
        if self.train_for_diff:
            n_obs = obs + out
        else:
            n_obs = out
        
        # replace obs buf with new obs
        self.env.obs_buf = n_obs
        
        # reward
        rewards = self.agent._calc_disc_rewards(n_obs)
        
        # terminal condition (currently set up for this -- should I use the actual env to determine this?)
        done = self.horizon_done(n_obs) or self.norm_done(n_obs)
        
        return n_obs, rewards, done, {}
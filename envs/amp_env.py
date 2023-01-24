import isaacgym
from isaacgymenvs.tasks.humanoid_amp import HumanoidAMP
from milo.dynamics_models.ensembles import DynamicsEnsemble
import numpy as np
import gym
import torch

class AMPHumanoidWrapper(gym.Wrapper):
    def __init__(self,
                 env: HumanoidAMP,
                 ensemble: DynamicsEnsemble,
                 init_state_buffer=None,
                 norm_thresh=float('inf')):
        super().__init__(env)
        self.ensemble = ensemble
        self.horizon = env.max_episode_length
        
        # additional env things
        self.ob = None
        self.num_steps = 0
        self.reset_counter = 0 # for the model selection
        
        # current dynamics model
        self.curr_model = ensemble.models[0]
        self.curr_model.eval()
        
        # init state buffer
        self.init_state_buffer = init_state_buffer
        
        # norm threshold so we don't like go into a weird area of the env
        self.norm_thresh = norm_thresh
        
    def is_done(self):
        horizon_done = self.num_steps >= self.horizon
        norm_done = np.linalg.norm(self.ob) >= self.norm_thresh
        env_done = self.env.get_done(self.ob)
        
        return bool(horizon_done or norm_done or env_done)
    
    def step(self, actions: torch.Tensor):
        pass
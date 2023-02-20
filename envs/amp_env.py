import isaacgym
from isaacgymenvs.tasks.humanoid_amp import HumanoidAMP
from milo.dynamics_models.ensembles import DynamicsEnsemble

from isaacgymenvs.learning.amp_players import AMPPlayerContinuous
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import torch
import numpy as np
from typing import Dict
from copy import deepcopy

class ModelBasedWrapper:
    def __init__(self,
                 env: HumanoidAMP,
                 cfg: Dict,
                 agent_chkpt: str,
                 ensemble: DynamicsEnsemble,
                 init_state_buffer=None,
                 norm_thresh = float('inf')):
        '''Model based wrapper around VecTasks.'''
        
        self.env = env
        self.ensemble = ensemble
        # fully in eval mode
        for model in self.ensemble.models:
            model = model.cuda()
            model.eval()
        
        self.train_for_diff = self.ensemble.models[0].train_for_diff
        self.horizon = env.max_episode_length
        
        # need to add trained amp agent to get the reward
        train_cfg = omegaconf_to_dict(cfg.train)
        # print(train_cfg)
        # {'params': {'seed': 42, 'algo': {'name': 'amp_continuous'}, 'model': {'name': 'continuous_amp'}, 'network': {'name': 'amp', 'separate': True, 'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None', 'mu_init': {'name': 'default'}, 'sigma_init': {'name': 'const_initializer', 'val': -2.9}, 'fixed_sigma': True, 'learn_sigma': False}}, 'mlp': {'units': [1024, 512], 'activation': 'relu', 'd2rl': False, 'initializer': {'name': 'default'}, 'regularizer': {'name': 'None'}}, 'disc': {'units': [1024, 512], 'activation': 'relu', 'initializer': {'name': 'default'}}}, 'load_checkpoint': False, 'load_path': '', 'config': {'name': 'HumanoidAMP', 'full_experiment_name': 'HumanoidAMP', 'env_name': 'rlgpu', 'ppo': True, 'multi_gpu': False, 'mixed_precision': False, 'normalize_input': True, 'normalize_value': True, 'value_bootstrap': True, 'num_actors': 4096, 'reward_shaper': {'scale_value': 1}, 'normalize_advantage': True, 'gamma': 0.99, 'tau': 0.95, 'learning_rate': 5e-05, 'lr_schedule': 'constant', 'kl_threshold': 0.008, 'score_to_win': 20000, 'max_epochs': 5000, 'save_best_after': 100, 'save_frequency': 50, 'print_stats': True, 'grad_norm': 1.0, 'entropy_coef': 0.0, 'truncate_grads': False, 'e_clip': 0.2, 'horizon_length': 16, 'minibatch_size': 32768, 'mini_epochs': 6, 'critic_coef': 5, 'clip_value': False, 'seq_len': 4, 'bounds_loss_coef': 10, 'amp_obs_demo_buffer_size': 200000, 'amp_replay_buffer_size': 1000000, 'amp_replay_keep_prob': 0.01, 'amp_batch_size': 512, 'amp_minibatch_size': 4096, 'disc_coef': 5, 'disc_logit_reg': 0.05, 'disc_grad_penalty': 0.2, 'disc_reward_scale': 2, 'disc_weight_decay': 0.0001, 'normalize_amp_input': True, 'task_reward_w': 0.0, 'disc_reward_w': 1.0}}}
        params = deepcopy(train_cfg['params'])
        # params = deepcopy(cfg['params'])
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
        
    def reset(self) -> torch.Tensor:
        # reset model by choosing the next one in the ensemble
        self.num_steps = 0
        self.model_idx = (self.model_idx + 1) % self.ensemble.n_models
        
        # reset init state with stuff from buffer every so often
        if self.init_state_buffer is not None:
            idx = np.random.randint(0, len(self.init_state_buffer))
            obs_dict = self.init_state_buffer[idx] if self.reset_counter % 5 == 0 else self.env.reset()
        else:
            obs_dict = self.env.reset()
            
        self.reset_counter += 1
        return obs_dict['obs']
    
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
        self.num_steps += 1
        
        # terminal condition (currently set up for this -- should I use the actual env to determine this?)
        done = self.horizon_done() or self.norm_done(n_obs)
        
        return n_obs, rewards, done, {}
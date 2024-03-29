import isaacgym
import isaacgymenvs
from isaacgymenvs.learning import amp_network_builder
from isaacgymenvs.learning import amp_models
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv

from rl_games.common import env_configurations, vecenv
from rl_games.algos_torch import model_builder

import torch
import hydra
from milo.dynamics_models.ensembles import DynamicsEnsemble
from datasets.dataset import *
from envs.amp_env import *
import gym
import os

'''Various testing methods.'''

def test_dataloading():
    '''test dataloading shapes'''
    stuff = torch.load('./data/offline/backflip/medium_dataset_all.pt', map_location='cpu')
    dataset = OfflineDataset(*stuff, n=3, device='cpu')
    dataloader = iterative_dataloader(dataset, 32)
    
    batch = next(iter(dataloader))
    s, a, r, ns, d = batch
    print(f'state shape: {s.size()}')
    print(f'action shape: {a.size()}')
    print(f'reward shape: {r.size()}')
    print(f'next state shape: {ns.size()}')
    print(f'done shape: {d.size()}')

@hydra.main(config_path='./milo_cfgs', config_name='config')
def test_penalty(cfg):
    
    random_data = torch.load('/home/ds844/amp_milo/data/offline/backflip/random_dataset.pt', map_location='cpu')
    medium_data = torch.load('/home/ds844/amp_milo/data/offline/backflip/medium_dataset.pt', map_location='cpu')
    
    random_states, random_actions, _ = random_data
    medium_states, medium_actions, _ = medium_data
    
    print('*' * 50)
    print('=' * 10 + ' SHAPES ' + '=' * 10)
    print(f'random: {random_states.size(), random_actions.size()}')
    print(f'medium: {medium_states.size(), medium_actions.size()}')
    print('*' * 50)
    
    trained_ensemble_diff = torch.load('/home/ds844/amp_milo/pretrained_dynamics_models/backflip/medium/densenet/ensemble_7_True_sgd_0.0003.pt', map_location='cpu')
    if isinstance(trained_ensemble_diff, DynamicsEnsemble):
        print('nice')
        trained_ensemble_diff.discrepancy_cfg.max_diff = False # to test log var estimates
    # trained_ensemble_state = torch.load('/home/ds844/amp_milo/pretrained_dynamics_models/backflip/medium/densenet/dynamics_ensemble_5_False.pt', map_location='cpu')
    
    # penalty on random data
    rand_diffs = []
    for i in range(random_states.size(0) // 100000):
        sbatch, abatch = random_states[i * 100000:(i + 1) * 100000], random_actions[i * 100000:(i + 1) * 100000]
        print(f'data sizes: {sbatch.size(), abatch.size()}')
        rand_pens_diff = trained_ensemble_diff.compute_onestep_discrepancy(sbatch, abatch, to_cpu=True)
        print(f'penalty sizes: {rand_pens_diff.size()}')
        rand_diffs.append(rand_pens_diff.mean())
    
    last_sbatch, last_abatch = random_states[(random_states.size(0) // 100000) * 100000:], random_actions[(random_states.size(0) // 100000) * 100000:]
    # print(last_sbatch.size(), last_abatch.size())
    rand_pens_diff_last = trained_ensemble_diff.compute_onestep_discrepancy(last_sbatch, last_abatch, to_cpu=True)
    rand_diffs.append(rand_pens_diff_last.mean())
    rand_diffs = torch.FloatTensor(rand_diffs)
    
    # rand_pens_state = trained_ensemble_state.compute_onestep_discrepancy(random_states, random_actions)
    print('---------------------------------------------------')
    
    # penalty on medium data
    med_diffs = []
    for i in range(medium_states.size(0) // 100000):
        sbatch, abatch = medium_states[i * 100000:(i + 1) * 100000], medium_actions[i * 100000:(i + 1) * 100000]
        print(sbatch.size(), abatch.size())
        med_pens_diff = trained_ensemble_diff.compute_onestep_discrepancy(sbatch, abatch, to_cpu=True)
        med_diffs.append(med_pens_diff.mean())
    
    last_sbatch, last_abatch = medium_states[(medium_states.size(0) // 100000) * 100000:], medium_actions[(medium_states.size(0) // 100000) * 100000:]
    print(last_sbatch.size(), last_abatch.size())
    med_pens_diff_last = trained_ensemble_diff.compute_onestep_discrepancy(last_sbatch, last_abatch, to_cpu=True)
    med_diffs.append(med_pens_diff_last.mean())
    med_diffs = torch.FloatTensor(med_diffs)
    # med_pens_state = trained_ensemble_state.compute_onestep_discrepancy(medium_states, medium_actions)
    
    print('=' * 20 + ' DIFF STATS ' + '=' * 20)
    print(f'Random penalty stats: {rand_diffs.min(), rand_diffs.max(), rand_diffs.mean(), rand_diffs.std()}')
    print(f'Medium penalty stats: {med_diffs.min(), med_diffs.max(), med_diffs.mean(), med_diffs.std()}')
    # print('=' * 20 + ' STATE STATS ' + '=' * 20)
    # print(f'Random penalty stats: {rand_pens_state.min(), rand_pens_state.max(), rand_pens_state.mean(), rand_pens_state.std()}')
    # print(f'Medium penalty stats: {med_pens_state.min(), med_pens_state.max(), med_pens_state.mean(), med_pens_state.std()}')
    print('=' * (40 + len(' STATE STATS ')))
    
def dataset_diffs():
    '''Tests to see how different the datasets themselves are from each other.'''
    rand_dataset = torch.load('/home/ds844/amp_milo/data/offline/backflip/random_dataset.pt', map_location='cpu')
    medium_dataset = torch.load('/home/ds844/amp_milo/data/offline/backflip/medium_dataset.pt', map_location='cpu')
    
    rand_state_mean, rand_state_std = torch.mean(rand_dataset[0]), torch.std(rand_dataset[0])
    med_state_mean, med_state_std = torch.mean(medium_dataset[0]), torch.std(medium_dataset[0])
    
    rand_action_mean, rand_action_std = torch.mean(rand_dataset[1]), torch.std(rand_dataset[1])
    med_action_mean, med_action_std = torch.mean(medium_dataset[1]), torch.mean(medium_dataset[1])
    
    print(f'Random state/action stats: \n state mean/std: {rand_state_mean, rand_state_std} \n action mean/std: {rand_action_mean, rand_action_std}')
    print(f'Medium state/action stats: \n state mean/std: {med_state_mean, med_state_std} \n action mean/std: {med_action_mean, med_action_std}')

@hydra.main(config_path='./amp_cfgs', config_name='config')
def test_mb_env(cfg):

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    cfg.sim_device = 'cpu' # because with CUDA you have to handle the tensor api, which may not be necessary right now
    print(f'=== devices: {cfg.sim_device, cfg.rl_device} ===')
    
    # set up env creator fn
    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/nothing_worth_noting",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs
    
    # set up RLGames vec env + rlgpu
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })
    
    # set up model builder to handle AMP model building
    model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
    # torch.backends.cudnn.benchmark = True
    
    env = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg
    )
    print('=== created env! ===')
    print(env.observation_space.shape, env.action_space.shape)
    assert isinstance(env, HumanoidAMP), "not an instance of humanoid amp env!"
    
    agent_chkpt = './runs/amp_backflip/nn/amp_backflip_5000.pth'
    ensemble = torch.load('./pretrained_dynamics_models/backflip/medium/densenet/ensemble_5_True_sgd_0.0003_False.pt', map_location=cfg.sim_device)
    print('=== loaded agent + ensemble checkpoint! ===')
    
    mb_env = ModelBasedWrapper(env, cfg, agent_chkpt, ensemble)
    print('=== created model based wrapper! ===')
    
    for _ in range(300):
        action = torch.randn((cfg.task.env.numEnvs,) + env.action_space.shape, device=cfg.sim_device)
        n_obs, r, done, _ = mb_env.step(action)
        
        print('output shapes')
        print(n_obs.size())
        print(r.size())
        print(done.size())
    
if __name__ == '__main__':
    test_mb_env()
import isaacgym
import gym
import isaacgymenvs
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_players
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder
from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver
from isaacgymenvs.utils.reformat import omegaconf_to_dict

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.algos_torch import model_builder
from datasets.dataset import AMPExpertDataset, OfflineDataset
import torch
import hydra

def build_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
    
    return runner

@hydra.main(config_path='./amp_cfgs', config_name='config')
def get_data(cfg):
    # set up env
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
            name = 'data_collection_all'
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs
    
    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })
    
    # set train / test config (i guess)
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    
    # init runner
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    
    # init checkpoint
    # checkpoint = './runs/amp_backflip/nn/amp_backflip_50.pth'
    # checkpoint = './runs/amp_backflip/nn/amp_backflip_500.pth'
    checkpoint = './runs/amp_backflip/nn/amp_backflip_5000.pth'
    
    if checkpoint == './runs/amp_backflip/nn/amp_backflip_50.pth':
        save_path = './data/offline/backflip/random_dataset_all.pt'
        total_games = 30000
    elif checkpoint == './runs/amp_backflip/nn/amp_backflip_500.pth':
        save_path = './data/offline/backflip/medium_dataset_all.pt'
        total_games = 3000
    else:
        save_path = './data/expert/backflip/expert_dataset_all.pt'
        total_games = 2000
    
    if checkpoint == './runs/amp_backflip/nn/amp_backflip_50.pth':
        print('====== GETTING RANDOM DATASET ======')
    elif checkpoint == './runs/amp_backflip/nn/amp_backflip_500.pth':
        print('====== GETTING MEDIUM DATASET ======')
    else:
        print('====== GETTING EXPERT DATASET ======')
    
    # eval runner (define cfg + run)
    run_cfg = {
        'train': False,
        'play': True,
        'checkpoint': checkpoint,
        'log_data': True,
        'total_games': total_games,
        'save_path': save_path,
        'sigma': None
    }
    print('=' * 50)
    print(run_cfg)
    print('=' * 50)
    out = runner.run(run_cfg)
    print('-' * 50)
    print(f'output: {out}')
    if out is not None:
        if run_cfg['offline']:
            print('====== SAVING OFFLINE DATASET ======')
            state, action, next_state = out
            print('*' * 20)
            print(state.size(), action.size(), next_state.size())
            print('*' * 20)
            dataset = OfflineDataset(state, action, next_state, device='cpu')
            torch.save(dataset, f'./data/offline/backflip/{"random" if "_50." in checkpoint else "medium"}_dataset.pt')
        else:
            print('====== SAVING EXPERT DATASET ======')
            state, next_state = out
            print('*' * 20)
            print(state.size(), next_state.size())
            print('*' * 20)
            dataset = AMPExpertDataset(state, next_state, device='cpu')
            torch.save(dataset, './data/expert/backflip/expert_dataset.pt')

if __name__ == '__main__':
    get_data()
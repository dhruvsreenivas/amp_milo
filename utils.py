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
    
    # define checkpoints
    
    # set train / test config (i guess)
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    
    # init runner
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    
    # init checkpoint
    # checkpoint = './runs/amp_backflip/nn/amp_backflip_50.pth'
    # checkpoint = './runs/amp_backflip/nn/amp_backflip_800.pth'
    checkpoint = './runs/amp_backflip/nn/amp_backflip.pth'
    
    # save_path = './data/offline/backflip/random_dataset.pt'
    # save_path = './data/offline/backflip/medium_dataset.pt'
    save_path = './data/expert/backflip/expert_dataset.pt'
    
    # total_games = 6000
    # total_games = 2000
    total_games = 500
    
    # eval runner (define cfg + run)
    run_cfg = {
        'train': False,
        'play': True,
        'checkpoint': checkpoint,
        'log_data': True,
        'total_games': total_games,
        'offline': "backflip.pth" not in checkpoint,
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
            torch.save(dataset, f'./data/offline/backflip/{"medium" if "800" in checkpoint else "random"}_dataset.pt')
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
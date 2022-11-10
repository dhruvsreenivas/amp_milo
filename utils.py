import torch

from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_players
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder
from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver

from datasets.dataset import AMPExpertDataset, OfflineDataset

def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

if __name__ == '__main__':
    # init runner
    runner = build_runner(RLGPUAlgoObserver())
    
    # define checkpoints
    random_checkpoint = './runs/nn/HumanoidAMP_50.pth'
    medium_checkpoint = './runs/nn/HumanoidAMP_300.pth'
    expert_checkpoint = './runs/nn/HumanoidAMP.pth'
    
    for checkpoint in [random_checkpoint, medium_checkpoint, expert_checkpoint]:
        # test runner (define cfg + run)
        run_cfg = {
            'train': False,
            'play': True,
            'checkpoint': checkpoint,
            'log_data': True,
            'offline': checkpoint != expert_checkpoint,
            'sigma': None
        }
        out = runner.run(run_cfg)
        if out is not None:
            if run_cfg['offline']:
                state, action, next_state = out
                dataset = OfflineDataset(state, action, next_state, device='cpu')
                torch.save(dataset, f'./data/offline/{"medium" if checkpoint == medium_checkpoint else "random"}.pt')
            else:
                state, next_state = out
                dataset = AMPExpertDataset(state, next_state, device='cpu')
                torch.save(dataset, './data/expert/expert_dataset.pt')
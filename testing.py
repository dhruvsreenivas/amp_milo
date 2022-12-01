import torch
import hydra

'''Various testing methods.'''

@hydra.main(config_path='./milo_cfgs', config_name='config')
def test_penalty(cfg):
    cfg.dynamics_training.discrepancy.max_diff = True
    
    random_data = torch.load('/home/ds844/amp_milo/data/offline/backflip/random_dataset.pt', map_location='cpu')
    medium_data = torch.load('/home/ds844/amp_milo/data/offline/backflip/medium_dataset.pt', map_location='cpu')
    
    random_states, random_actions, _ = random_data
    medium_states, medium_actions, _ = medium_data
    
    print('*' * 50)
    print('=' * 10 + ' SHAPES ' + '=' * 10)
    print(f'random: {random_states.size(), random_actions.size()}')
    print(f'medium: {medium_states.size(), medium_actions.size()}')
    print('*' * 50)
    
    trained_ensemble_diff = torch.load('/home/ds844/amp_milo/pretrained_dynamics_models/backflip/medium/dynamics_ensemble_5_True.pt', map_location='cpu')
    trained_ensemble_state = torch.load('/home/ds844/amp_milo/pretrained_dynamics_models/backflip/medium/dynamics_ensemble_5_False.pt', map_location='cpu')
    
    # penalty on random data
    rand_pens_diff = trained_ensemble_diff.compute_onestep_discrepancy(random_states, random_actions)
    rand_pens_state = trained_ensemble_state.compute_onestep_discrepancy(random_states, random_actions)
    
    # penalty on medium data
    med_pens_diff = trained_ensemble_diff.compute_onestep_discrepancy(medium_states, medium_actions)
    med_pens_state = trained_ensemble_state.compute_onestep_discrepancy(medium_states, medium_actions)
    
    print('=' * 20 + ' DIFF STATS ' + '=' * 20)
    print(f'Random penalty stats: {rand_pens_diff.min(), rand_pens_diff.max(), rand_pens_diff.mean(), rand_pens_diff.std()}')
    print(f'Medium penalty stats: {med_pens_diff.min(), med_pens_diff.max(), med_pens_diff.mean(), med_pens_diff.std()}')
    print('=' * 20 + ' STATE STATS ' + '=' * 20)
    print(f'Random penalty stats: {rand_pens_state.min(), rand_pens_state.max(), rand_pens_state.mean(), rand_pens_state.std()}')
    print(f'Medium penalty stats: {med_pens_state.min(), med_pens_state.max(), med_pens_state.mean(), med_pens_state.std()}')
    print('=' * (40 + len(' STATE STATS ')))
    
if __name__ == '__main__':
    test_penalty()
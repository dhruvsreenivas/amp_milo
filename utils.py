import torch

def get_samples(chkpt, env, n_samples, offline=False):
    agent = torch.load(chkpt)
    agent.vec_env = env
    
    if offline:
        states = []
        actions = []
        next_states = []
    else:
        states = []
        next_states = []
    
    samples = 0
    ob = env.reset()
    done = False
    while samples < n_samples:
        ob_tensor = torch.FloatTensor(ob)
        res_dict = agent.get_action_values(ob_tensor)
        action = res_dict['actions']
        
        n_ob, _, done, _ = agent.env_step(action)
        
        if offline:
            states.append(ob_tensor)
            actions.append(torch.FloatTensor(action))
            next_states.append(torch.FloatTensor(n_ob))
        else:
            states.append(ob_tensor)
            next_states.append(torch.FloatTensor(n_ob))
        
        if done:
            ob = env.reset()
        else:
            ob = n_ob
            
        samples += ob_tensor.size(0)
        
    states = torch.stack(states, dim=0)
    next_states = torch.stack(next_states, dim=0)
    if offline:
        actions = torch.stack(actions, dim=0)
        return states, actions, next_states
    
    return states, next_states
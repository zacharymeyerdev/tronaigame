import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import config
import train_agents

class DQN(nn.Module):
    def __init__(self, in_features, hidden_layer_size, number_of_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, number_of_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    @staticmethod
    def select_action(state, epsilon, policy_net):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float, device=config.device)
                return policy_net(state).max(1)[1].view(1, 1).item()
        else:
            return random.randrange(config.action_size)
        
    @staticmethod
    def train_model(policy_net, target_net, optimizer, experiences, gamma):
        if len(experiences) < config.batch_size:
            return

    states, actions, rewards, next_states, dones = zip(*train_agents.experiences)
    states = torch.tensor(states, dtype=torch.float, device=config.device)
    actions = torch.tensor(actions, device=config.device)
    rewards = torch.tensor(rewards, device=config.device)
    next_states = torch.tensor(next_states, dtype=torch.float, device=config.device)
    dones = torch.tensor(dones, dtype=torch.uint8, device=config.device)

    state_action_values = config.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_state_values = torch.zeros(config.batch_size, device=config.device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=config.device, dtype=torch.bool)
    non_final_next_states = torch.tensor([s for s in next_states if s is not None], dtype=torch.float, device=config.device)
    next_state_values[non_final_mask] = config.target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * config.gamma) * (1 - dones) + rewards
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    config.optimizer.zero_grad()
    loss.backward()
    config.optimizer.step()

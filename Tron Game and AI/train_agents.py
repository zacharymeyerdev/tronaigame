import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
from trongame import TronGame  # Your game environment
from dqn_agent import DQN  # Your DQN model
import config

# Define device for training (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
hidden_layer_size = 64  # or any other number based on your preference
num_episodes = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
replay_memory_size = 100000
batch_size = 20
gamma = 0.99  # Discount factor
learning_rate = 0.001
target_update = 10
state_size = 8
action_size = 4
learning_rate = 0.001

# Initialize policy network (DQN) and target network
policy_net = DQN(config.state_size, config.hidden_layer_size, config.action_size).to(device)
target_net = DQN(config.state_size, config.hidden_layer_size, config.action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target net will not be trained
agent = policy_net

# Set up replay memory
replay_memory = deque(maxlen=replay_memory_size)

# Optimizer
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

# Initialize game environment and DQN agent
game = TronGame()
state_size = game.state_size  # Adjust according to your game's state representation
action_size = game.action_size  # Adjust according to your game's action space
agent = DQN(state_size, hidden_layer_size, action_size)

#train model function
def train_model(policy_net, experiences, gamma, optimizer):
    states, actions, rewards, next_states, dones = zip(*experiences)

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.uint8).to(device)  # Use uint8 for boolean

    # Get Q values for current states
    state_action_values = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Compute Q values for next states using target network
    next_state_values = torch.zeros(batch_size).to(device)
    non_final_next_states = torch.tensor([s for s, d in zip(next_states, dones) if not d], dtype=torch.float).to(device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device, dtype=torch.bool)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) * (1 - dones) + rewards

    # Compute loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
epsilon = epsilon_start
for episode in range(num_episodes):
    state = game.reset()
    total_reward = 0

    while True:
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = agent.select_action(state)
        else:
            action = random.randrange(action_size)

        next_state, reward, done = game.step(action)
        total_reward += reward

        # Store the transition in replay memory
        replay_memory.append((state, action, reward, next_state, done))

        # Train the model if memory is sufficient
        if len(replay_memory) > batch_size:
            experiences = random.sample(replay_memory, batch_size)
            train_model(agent, experiences, gamma, optimizer)

        state = next_state

        if done:
            break

    # Update epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Update the target network
    if episode % target_update == 0:
        agent.update_target_network()

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# Save the trained model
torch.save(agent.state_dict(), 'tron_dqn_agent.pth')

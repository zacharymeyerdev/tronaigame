import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
from trongame import TronGame  # Your game environment
import config
from collections import deque
import torch
from dqn_agent import DQN, select_action, train_model  # Your DQN model
# Define device for training (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize DQN components
policy_net = DQN(config.state_size, config.hidden_layer_size, config.action_size).to(device)
target_net = DQN(config.state_size, config.hidden_layer_size, config.action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.learning_rate)
replay_memory = deque(maxlen=config.replay_memory_size)

# Initialize game environment
game = TronGame()

# Training loop
epsilon = config.epsilon_start
for episode in range(config.num_episodes):
    print(episode)
    state = game.reset()
    total_reward = 0

    while True:
        print("true working")
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = select_action(state, epsilon, policy_net)  # Assuming this method exists in dqn_agent.py
            print("not random")
        else:
            action = random.randrange(game.action_size)
            print("random")

        next_state, reward, done = game.step(action)
        total_reward += reward

        replay_memory.append((state, action, reward, next_state, done))

        if len(replay_memory) > config.batch_size:
            print("training")
            experiences = random.sample(replay_memory, config.batch_size)
            train_model(policy_net, target_net, optimizer, experiences, config.gamma)

        state = next_state

        if done:
            print("done")
            break

    epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    if episode % config.target_update == 0:
        print("updating")
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

torch.save(policy_net.state_dict(), 'tron_dqn_agent.pth')
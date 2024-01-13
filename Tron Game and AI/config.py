import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
from trongame import TronGame  # Ensure this script provides necessary functionalities
from trongame import TronGame  # Your game environment
from dqn_agent import DQN  # Your DQN model

num_episodes = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
replay_memory_size = 100000
batch_size = 20
state_size = 8
action_size = 4
learning_rate = 0.001
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
TARGET_UPDATE = 10  # Update target network every 10 episodes
hidden_layer_size = 64  # Number of neurons in each hidden layer
# Define device for training (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize policy network (DQN) and target network
policy_net = DQN(state_size, hidden_layer_size, action_size).to(device)
target_net = DQN(state_size, hidden_layer_size, action_size).to(device)
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
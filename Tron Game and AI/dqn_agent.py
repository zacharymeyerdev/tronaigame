import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
from trongame import TronGame  # Ensure this script provides necessary functionalities

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

# Define the size of the input layer and the number of actions
# Assuming a grid size of 20x20 and simple state representation
grid_size = 20
in_features = 4 + 4  # Snake's head position (x, y), Food's position (x, y), Direction (one-hot encoded)
number_of_actions = 4  # Up, Down, Left, Right


# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
replay_memory_size = 100000
batch_size = 20
TARGET_UPDATE = 10  # Update target network every 10 episodes
hidden_layer_size = 64  # Number of neurons in each hidden layer


# Neural Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(in_features, hidden_layer_size, number_of_actions).to(device)
target_net = DQN(in_features, hidden_layer_size, number_of_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target net not used for training

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Replay Memory
replay_memory = deque(maxlen=replay_memory_size)

# Initialize your game environment
game = TronGame()

def train_model(policy_net, target_net, optimizer, replay_memory, batch_size, gamma):
    if len(replay_memory) < batch_size:
        return

    # Sample a batch from replay memory
    mini_batch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float).to(device)
    dones = torch.tensor(dones).to(device)

    # Get Q values for current states
    state_action_values = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Compute Q values for next states
    next_state_values = target_net(next_states).max(1)[0].detach()
    expected_state_action_values = rewards + (gamma * next_state_values * (1 - dones))

    # Compute loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def select_action(state, epsilon, policy_net):
    print(f"Selecting action, state: {state}, epsilon: {epsilon}")
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float).to(device)
            action = policy_net(state).max(1)[1].view(1, 1).item()
            print(f"Chosen action from model: {action}")
            return action
    else:
        action = random.randrange(number_of_actions)
        print(f"Chosen random action: {action}")
        return action

num_episodes = 1000  # Number of episodes to train on
hidden_layer_size = 64  # Example size, adjust as needed

for episode in range(num_episodes):
    state = game.reset()
    print(f"Episode {episode} started, initial state: {state}")
    total_reward = 0

    while True:
        action = select_action(state, epsilon, policy_net)
        next_state, reward, done = game.step(action)
        total_reward += reward

        replay_memory.append((state, action, reward, next_state, done))

        state = next_state

        train_model(policy_net, target_net, optimizer, replay_memory, batch_size, gamma)

        if done:
            break

    # Corrected target network update condition
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}")
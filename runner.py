import torch
from trongame import TronGame
from dqn_agent import DQN
import train_agents
import config
from train_agents import policy_net, target_net, optimizer, replay_memory

# To play the game with DQN
game = TronGame()
game.play(policy_net, target_net, optimizer, replay_memory, 
          config.epsilon_start, config.epsilon_decay, 
          config.epsilon_min, config.batch_size, config.gamma)
import numpy as np
import random
import os

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95):
        self.q_table = np.zeros((n_states, n_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.n_actions = n_actions

    def choose_action(self, state):
        # Ensure the state is within the valid range
        if state < 0 or state >= len(self.q_table):
            print(f"Invalid state index: {state}")
            # Handle the invalid state, e.g., clamp to the valid range
            state = max(min(state, len(self.q_table) - 1), 0)

        action_index = np.argmax(self.q_table[state])
        return self.actions[action_index]


    def learn(self, state, action, reward, next_state):
        state = int(state)  # Convert state to integer
        action_index = self.actions.index(action)

        # Retrieve current Q value
        current_q = self.q_table[state, action_index]

        # Retrieve maximum Q value for next state
        max_future_q = np.max(self.q_table[next_state])

        # Check if current_q, reward, and max_future_q are scalars
        if not np.isscalar(current_q):
            current_q = current_q[0]  # or however you want to handle it
        if not np.isscalar(reward):
            reward = reward[0]  # or however you want to handle it
        if not np.isscalar(max_future_q):
            max_future_q = max_future_q[0]  # or however you want to handle it

        # Calculate new Q value
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)

            # Update Q-table
        self.q_table[state, action_index] = new_q
        # Retrieve maximum Q value for next state

        # Check if max_future_q is a scalar
        if not np.isscalar(max_future_q):
            print(f"max_future_q is not a scalar: {max_future_q}")
        # Check if new_q is a scalar
        if not np.isscalar(new_q):
            print(f"new_q is not a scalar: {new_q}")

        # Debug prints
        print(f"State: {state}, Type: {type(state)}")
        print(f"Action Index: {action_index}, Type: {type(action_index)}")
        print(f"New Q-value: {new_q}, Type: {type(new_q)}")
        print(f"Reward: {reward}, Type: {type(reward)}")

        # Print the direction of the player avatar's movement
        if action == 'UP':
            print("Player avatar moves UP")
        elif action == 'DOWN':
            print("Player avatar moves DOWN")
        elif action == 'LEFT':
            print("Player avatar moves LEFT")
        elif action == 'RIGHT':
            print("Player avatar moves RIGHT")

    def save_q_table(self, filename):
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}.npy")

    def load_q_table(self, filename):
        if os.path.exists(filename + '.npy'):
            self.q_table = np.load(filename + '.npy')
            print(f"Q-table loaded from {filename}.npy")
        else:
            print("File not found. Ensure the correct path and filename.")
# Define the number of states and actions
n_states = 100  # Adjust based on your game's state representation
n_actions = 4   # UP, DOWN, LEFT, RIGHT

q_learning_agent = QLearningAgent(n_states, n_actions)

number_of_episodes = 1000 # Number of episodes for training
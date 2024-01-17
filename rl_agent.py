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

    def choose_action(self, state, previous_direction):
        print("State:", state, "Type:", type(state))
        
        # Define opposite directions
        opposite_directions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}

        # If no previous direction, proceed with original action selection process
        if previous_direction is None:
            if random.uniform(0, 1) < 0.05:  # Exploration factor
                return random.choice(self.actions)
            else:
                action_index = np.argmax(self.q_table[state])
                return self.actions[action_index]

        # If there is a previous direction, avoid choosing the opposite direction
        if random.uniform(0, 1) < 0.05:  # Exploration factor
            action = random.choice(self.actions)
            while action == opposite_directions[previous_direction]:
                action = random.choice(self.actions)
            return action
        else:
            action_indices = np.argsort(self.q_table[state])[::-1]
            for action_index in action_indices:
                action = self.actions[action_index]
                if action != opposite_directions[previous_direction]:
                    return action

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
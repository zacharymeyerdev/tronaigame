import pygame
import random
import config
import torch.optim as optim
from collections import deque
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
    
# Correct Initialization of DQN models

class TronGame:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.avatar_size = 10
        self.ai1_avatar_speed = 10
        self.ai2_avatar_speed = 10
        self.clock = pygame.time.Clock()
        self.game_display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Tron Clone')
        self.reset()

    def reset(self):
        # Reset the game to its initial state
        self.ai1_avatar_x = 3 * self.width / 4
        self.ai1_avatar_y = self.height / 2
        self.ai2_avatar_x = self.width / 4
        self.ai2_avatar_y = self.height / 2
        self.ai1_last_direction = ''
        self.ai2_last_direction = ''
        self.ai1_trails = []
        self.ai2_trails = []
        return (self.ai1_avatar_x, self.ai1_avatar_y, 
            self.ai2_avatar_x, self.ai2_avatar_y)

    def step(self, action):
        # Assume action is a direction change for AI1
        # Update AI1
        self.ai1_avatar_x, self.ai1_avatar_y, self.ai1_last_direction, self.ai1_trails = self.update_ai_avatar(
            self.ai1_avatar_x, self.ai1_avatar_y, self.ai1_avatar_speed, action, self.ai1_trails)

        # Update AI2
        self.ai2_avatar_x, self.ai2_avatar_y, self.ai2_last_direction, self.ai2_trails = self.update_ai_avatar(
            self.ai2_avatar_x, self.ai2_avatar_y, self.ai2_avatar_speed, self.ai2_last_direction, self.ai2_trails)

        # Check for collisions
        game_over = self.is_collision(self.ai1_avatar_x, self.ai1_avatar_y, self.ai1_trails) or \
                    self.is_collision(self.ai1_avatar_x, self.ai1_avatar_y, self.ai2_trails) or \
                    self.is_collision(self.ai2_avatar_x, self.ai2_avatar_y, self.ai2_trails) or \
                    self.is_collision(self.ai2_avatar_x, self.ai2_avatar_y, self.ai1_trails)

        # Determine the reward
        reward = 0.01
        if game_over:
            reward = -1

        # New state could be positions and directions of avatars
        new_state = (self.ai1_avatar_x, self.ai1_avatar_y, self.ai1_last_direction, 
                    self.ai2_avatar_x, self.ai2_avatar_y, self.ai2_last_direction)

        return new_state, reward, game_over

    def render(self):
        # Draw the avatars and trails
        self.game_display.fill((255, 255, 255))  # White background
        pygame.draw.rect(self.game_display, (0, 0, 0), [self.ai2_avatar_x, self.ai2_avatar_y, self.avatar_size, self.avatar_size])
        pygame.draw.rect(self.game_display, (0, 0, 0), [self.ai1_avatar_x, self.ai1_avatar_y, self.avatar_size, self.avatar_size])

        for trail in self.ai2_trails:
            pygame.draw.rect(self.game_display, (0, 0, 255), [trail[0], trail[1], self.avatar_size, self.avatar_size])

        for trail in self.ai1_trails:
            pygame.draw.rect(self.game_display, (255, 165, 0), [trail[0], trail[1], self.avatar_size, self.avatar_size])
        
        pygame.display.update()

    def is_collision(self, ai_x, ai_y, trails):
        for trail in trails:
            if ai_x == trail[0] and ai_y == trail[1]:
                return True
        return False
    
    def update_ai_avatar(self, ai_x, ai_y, ai_speed, last_direction, trails):
        # Choose a new direction that is not the reverse of the current direction
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        if last_direction == 'UP':
            directions.remove('DOWN')
        elif last_direction == 'DOWN':
            directions.remove('UP')
        elif last_direction == 'LEFT':
            directions.remove('RIGHT')
        elif last_direction == 'RIGHT':
            directions.remove('LEFT')

        new_direction = random.choice(directions)

        # Update the position based on the new direction
        if new_direction == 'UP':
            ai_y -= ai_speed
        elif new_direction == 'DOWN':
            ai_y += ai_speed
        elif new_direction == 'LEFT':
            ai_x -= ai_speed
        elif new_direction == 'RIGHT':
            ai_x += ai_speed

        # Ensure the avatar stays within the game boundaries
        ai_x = max(0, min(self.width, ai_x))
        ai_y = max(0, min(self.height, ai_y))

        # Add the new position to the trail
        trails.append((ai_x, ai_y))

        # Return the updated position, direction, and trail
        return ai_x, ai_y, new_direction, trails
    
    def draw_avatars(self):
        # Draw the avatars and trails
        self.game_display.fill((255, 255, 255))  # White background
        pygame.draw.rect(self.game_display, (0, 0, 0), [self.ai1_avatar_x, self.ai1_avatar_y, self.avatar_size, self.avatar_size])
        pygame.draw.rect(self.game_display, (0, 0, 0), [self.ai2_avatar_x, self.ai2_avatar_y, self.avatar_size, self.avatar_size])

        for trail in self.ai1_trails:
            pygame.draw.rect(self.game_display, (0, 0, 255), [trail[0], trail[1], self.avatar_size, self.avatar_size])

        for trail in self.ai2_trails:
            pygame.draw.rect(self.game_display, (255, 165, 0), [trail[0], trail[1], self.avatar_size, self.avatar_size])

        pygame.display.update()
        
    def play(self, policy_net, target_net, optimizer, replay_memory, epsilon_start, epsilon_decay, epsilon_min, batch_size, gamma):
        epsilon = epsilon_start
        replay = True
        import dqn_agent

        while replay:
            game_over = False
            state = self.reset()  # Get the initial state
            if state is None:
                raise ValueError("Received 'None' state from reset")
            total_reward = 0

            while not game_over:
                # Nested function for selecting an action

                # Nested function for training the model
                    
                action = dqn_agent.select_action(state, epsilon, policy_net)  # DQN agent selects an action
                next_state, reward, game_over = self.step(action)  # Apply action to the game

                # Store the transition in replay memory
                total_reward += reward
                replay_memory.append((state, action, reward, next_state, game_over))

                # Train the model if memory is sufficient
                if len(replay_memory) > batch_size:
                    experiences = random.sample(replay_memory, batch_size)
                    dqn_agent.train_model(policy_net, target_net, optimizer, experiences, gamma)

                state = next_state  # Update the state
                
                if game_over:
                    print(f"Game over at this step. Total reward: {total_reward}")
            
                # Update epsilon for the epsilon-greedy strategy
                epsilon = max(epsilon_min, epsilon_decay * epsilon)

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_over = True
                        replay = False

                # Render the game
                self.draw_avatars()
                self.clock.tick(60)  # Control the frame rate

            # Reset the game state for the next episode
            self.reset()
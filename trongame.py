import pygame
import sys
import random
from rl_agent import QLearningAgent
import pygame.font

# Initialize Pygame
pygame.init()
pygame.font.init()

# Game Variables

# Initialize the font object
font = pygame.font.Font(None, 36)
width, height = 1200, 800
player_size = 10
player1_pos = [width // 4, height // 2]
player2_pos = [3 * width // 4, height // 2]
player1_color = (0, 0, 255)  # Blue
player1_trail_color = (0, 128, 255)  # Dark Blue
player2_color = (255, 100, 0)  # Orange
player2_trail_color = (255, 200, 0)  # Dark Orange
bg_color = (255, 255, 255)  # White
player1_trail = []
player2_trail = []
player1_direction = 'RIGHT'
player2_direction = 'LEFT'
speed = 10
player1_previous_direction = None
player2_previous_direction = None

# Instantiate two QLearningAgents
n_states = 100  # Adjust based on your game's state representation
n_actions = 4   # UP, DOWN, LEFT, RIGHT
q_learning_agent1 = QLearningAgent(n_states, n_actions)
q_learning_agent2 = QLearningAgent(n_states, n_actions)

# Set up the display
screen = pygame.display.set_mode((width, height))
screen.fill(bg_color)
pygame.display.set_caption("Tron Game")
clock = pygame.time.Clock()
pygame.display.flip()

def choose_ai_action():
    # Random action selector for the enemy
    return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])

def update_ai():
    global player1_direction, player2_direction, player1_previous_direction, player2_previous_direction
    action1 = q_learning_agent1.choose_action(get_current_state_player1(), player1_previous_direction)
    action2 = q_learning_agent2.choose_action(get_current_state_player2(), player2_previous_direction)
    player1_direction = action1
    player2_direction = action2
    pygame.display.update()

def draw_trails():
    for pos in player1_trail:
        pygame.draw.rect(screen, player1_trail_color, (pos[0], pos[1], player_size, player_size))
    for pos in player2_trail:
        pygame.draw.rect(screen, player2_trail_color, (pos[0], pos[1], player_size, player_size))

def check_collisions_player1():
    # Check border collisions for player 1
    if player1_pos[0] >= width or player1_pos[0] < 0 or player1_pos[1] >= height or player1_pos[1] < 0:
        return True

    # Check trail collisions for player 1
    if player1_pos in player1_trail[:-1] or player1_pos in player2_trail:
        return True

    return False

def check_collisions_player2():
    # Check border collisions for player 2
    if player2_pos[0] >= width or player2_pos[0] < 0 or player2_pos[1] >= height or player2_pos[1] < 0:
        return True

    # Check trail collisions for player 2
    if player2_pos in player2_trail[:-1] or player2_pos in player1_trail:
        return True

    return False

def calculate_distance(pos1, pos2):
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def update_reward_for_proximity():
    distance = calculate_distance(player1_pos, player2_pos)
    if distance < 500:
        proximity_reward = 1 / (distance + 1)  # Adding 1 to avoid division by zero
        proximity_reward *= 100  # Increase the proximity reward
        return proximity_reward
    else:
        return 0  # No reward if distance is 500 or more

def update_reward_for_distance():
    distance = calculate_distance(player1_pos, player2_pos)
    if distance > 500:
        distance_reward = 1 / (distance + 1)  # Adding 1 to avoid division by zero
        distance_reward *= -200  # Increase the proximity penalty
        return distance_reward
    else:
        return 0  # No penalty if distance is 500 or less


def reset_game():
    # Reset the game state to its initial conditions
    global player1_pos, player2_pos, player1_trail, player2_trail, game_over
    player1_pos = [width // 4, height // 2]
    player2_pos = [3 * width // 4, height // 2]
    player1_trail = []
    player2_trail = []
    game_over = False
    # Any other necessary resets

def get_current_state_player1():
    # Calculate the state representation for player 1
    state_player1 = (player1_pos[0] // (width // 10)) + (player1_pos[1] // (height // 10)) * 10
    return state_player1

def get_current_state_player2():
    # Calculate the state representation for player 2
    state_player2 = (player2_pos[0] // (width // 10)) + (player2_pos[1] // (height // 10)) * 10
    return state_player2

def draw_players():
    pygame.draw.rect(screen, player1_color, (player1_pos[0], player1_pos[1], player_size, player_size))
    pygame.draw.rect(screen, player2_color, (player2_pos[0], player2_pos[1], player_size, player_size))

def update_positions_player1():
    global player1_pos, player1_trail, player1_previous_direction, player1_direction

    # Update the previous direction
    player1_previous_direction = player1_direction

    # Update Player 1 Position
    if player1_direction == 'UP':
        player1_pos[1] -= speed
    elif player1_direction == 'DOWN':
        player1_pos[1] += speed
    elif player1_direction == 'LEFT':
        player1_pos[0] -= speed
    elif player1_direction == 'RIGHT':
        player1_pos[0] += speed

    # Add the current position to Player 1's trail
    player1_trail.append(list(player1_pos))
    pygame.display.update()

def update_positions_player2():
    global player2_pos, player2_trail, player2_previous_direction, player2_direction

    # Update the previous direction
    player2_previous_direction = player2_direction

    # Update Player 2 Position
    if player2_direction == 'UP':
        player2_pos[1] -= speed
    elif player2_direction == 'DOWN':
        player2_pos[1] += speed
    elif player2_direction == 'LEFT':
        player2_pos[0] -= speed
    elif player2_direction == 'RIGHT':
        player2_pos[0] += speed

    # Add the current position to Player 2's trail
    player2_trail.append(list(player2_pos))
    pygame.display.update()


def perform_action_player1(action):
    global player1_pos, player1_trail, game_over
    reward1 = 0.1  # Initialize reward for player 1

    # Apply action for player 1
    # Update player 1 position based on action
    # Update player 1 trail
    proximity_reward = update_reward_for_proximity()
    distance_reward = update_reward_for_distance()
    reward1 += proximity_reward  # Encourage moving closer
    reward1 += distance_reward
    # Check collisions for player 1
    game_over1 = check_collisions_player1()

    if game_over1:
        reward1 -= 1  # Subtract 1 from reward for going into wall
    elif player1_pos in player1_trail:
        reward1 -= 1.5  # Subtract 1.5 from reward for going into own trail
    elif player1_pos in player2_trail:
        reward1 -= 0.5  # Add 2 to reward for going into opponent's trail

    return reward1, game_over1

def perform_action_player2(action):
    global player2_pos, player2_trail, game_over
    reward2 = 0.1  # Initialize reward for player 2

    # Apply action for player 2
    # Update player 2 position based on action
    # Update player 2 trail
    proximity_reward = update_reward_for_proximity()
    distance_reward = update_reward_for_distance()
    reward2 += proximity_reward  # Encourage moving closer
    reward2 += distance_reward
    # Check collisions for player 2
    game_over2 = check_collisions_player2()

    if game_over2:
        reward2 -= 1  # Subtract 1 from reward for going into wall
    elif player2_pos in player2_trail:
        reward2 -= 1.5  # Subtract 1.5 from reward for going into own trail
    elif player2_pos in player1_trail:
        reward2 -=0.5  # Add 2 to reward for going into opponent's trail

    return reward2, game_over2

def update_positions():
    global player1_pos, player2_pos, player1_trail, player2_trail, player1_direction, player2_direction

    # Update Player 1 Position
    if player1_direction == 'UP':
        player1_pos[1] -= speed
    elif player1_direction == 'DOWN':
        player1_pos[1] += speed
    elif player1_direction == 'LEFT':
        player1_pos[0] -= speed
    elif player1_direction == 'RIGHT':
        player1_pos[0] += speed

    # Update Player 2 Position
    if player2_direction == 'UP':
        player2_pos[1] -= speed
    elif player2_direction == 'DOWN':
        player2_pos[1] += speed
    elif player2_direction == 'LEFT':
        player2_pos[0] -= speed
    elif player2_direction == 'RIGHT':
        player2_pos[0] += speed

    # Add the current position to the trails
    player1_trail.append(list(player1_pos))
    player2_trail.append(list(player2_pos))
    
# Assuming number_of_episodes is defined
number_of_episodes = 1000
def play_game_with_agents(agent1, agent2, episode_number):
    global game_over
    print("Starting Episode", episode_number)
    reset_game()
    game_over = False
    state1 = get_current_state_player1()  # Define this function based on player 1's perspective
    state2 = get_current_state_player2()  # Define this function based on player 2's perspective
    draw_players()  # Draw the players at their updated positions
    pygame.display.flip()
    while not game_over:
        # Draw the game state

        draw_trails()  # Draw the trails of both players
        draw_players()  # Draw the players at their updated positions
        
        # Display the episode number
        episode_text = font.render(f'Episode: {episode_number}', True, (0, 0, 0))  # Black color for the text
        screen.blit(episode_text, (width - 200, 20))  # Position the text

        pygame.display.flip()

        # Handle events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Agent 1's turn
        action1 = agent1.choose_action(state1, player1_previous_direction)
        update_ai()  # Update the AI's actions
        reward1 = perform_action_player1(action1)  # Define this function for player 1's actions
        new_state1 = get_current_state_player1()
        agent1.learn(state1, action1, reward1, new_state1)
        state1 = new_state1

        # Agent 2's turn
        action2 = agent2.choose_action(state2, player2_previous_direction)
        update_ai()
        reward2 = perform_action_player2(action2)  # Define this function for player 2's actions
        new_state2 = get_current_state_player2()
        agent2.learn(state2, action2, reward2, new_state2)
        state2 = new_state2

        # Draw the game state
        screen.fill(bg_color)
        draw_trails()  # Draw the trails of both players
        draw_players()  # Draw the players at their updated positions
        pygame.display.flip()

        pygame.time.delay(100)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Check if the game is over
        if check_collisions_player1() or check_collisions_player2():
            game_over = True  # Restart the game
            state1 = get_current_state_player1()
            state2 = get_current_state_player2()
            continue

        clock.tick(10)

        # Update positions
        update_positions_player1()
        update_positions_player2()

        # Update states
        state1 = get_current_state_player1()
        state2 = get_current_state_player2()

        if episode_number % 100 == 0:
            q_learning_agent1.save_q_table(f'q_table_agent1_episode_{episode_number}')
            q_learning_agent2.save_q_table(f'q_table_agent2_episode_{episode_number}')

    print("Game Over")

for _ in range(100):
    q_learning_agent1.load_q_table('final_q_table_agent1')
    q_learning_agent2.load_q_table('final_q_table_agent2')
    for episode in range(number_of_episodes):
        play_game_with_agents(q_learning_agent1, q_learning_agent2, episode_number=episode + 1)
        print(f"Episode {episode + 1} completed")
    
    q_learning_agent1.save_q_table('final_q_table_agent1')
    q_learning_agent2.save_q_table('final_q_table_agent2')

# After training
#print("beans1")
#train_ai()  # Comment this out after training is done
#print("beans2")
# Optionally, save the Q-tables at the end of training
# To play the game
#game_loop(training_mode=True)  # Uncomment this to play the game

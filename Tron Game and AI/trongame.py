import pygame
import time
import random

# Initialize Pygame
pygame.init()

# define ai_last_direction
ai_last_direction = 'UP'

# Game window dimensions
width, height = 800, 600
box_margin = 50
avatar_size = 10

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
orange = (255, 165, 0)

# Create the game window
game_display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tron Clone')
 
# Avatar properties
player_avatar_speed = 15
ai_avatar_speed = 10

# Initial avatar positions
player_avatar_x = width / 4
player_avatar_y = height / 2
ai_avatar_x = 3 * width / 4
ai_avatar_y = height / 2

player_avatar_x_change = 0
player_avatar_y_change = 0

# Box properties
box_margin = 50
box_color = red

# Clock
clock = pygame.time.Clock()

# Trails
player_trails = []
ai_trails = []

def check_collision():
    global player_avatar_x, player_avatar_y, player_trails

    for i in range(len(player_trails)):
        trail = player_trails[i]
        if player_avatar_x == trail[0] and player_avatar_y == trail[1]:
            if i < len(player_trails) - 3:
                player_trails = player_trails[:i+1]
            else:
                player_trails = player_trails[:i]
            break

def update_avatars():
    global player_avatar_x, player_avatar_y, ai_avatar_x, ai_avatar_y, player_trails, ai_trails, ai_last_direction

    # Player avatar update
    player_avatar_x += player_avatar_x_change
    player_avatar_y += player_avatar_y_change

    # Add current position to player trail
    player_trails.append((player_avatar_x, player_avatar_y))

    # AI avatar update
    # Remove the reverse of the current direction from choices
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    if ai_last_direction == 'UP':
        directions.remove('DOWN')
    elif ai_last_direction == 'DOWN':
        directions.remove('UP')
    elif ai_last_direction == 'LEFT':
        directions.remove('RIGHT')
    elif ai_last_direction == 'RIGHT':
        directions.remove('LEFT')

    ai_avatar_direction = random.choice(directions)
    ai_last_direction = ai_avatar_direction

    if ai_avatar_direction == 'UP':
        ai_avatar_y -= ai_avatar_speed
    elif ai_avatar_direction == 'DOWN':
        ai_avatar_y += ai_avatar_speed
    elif ai_avatar_direction == 'LEFT':
        ai_avatar_x -= ai_avatar_speed
    elif ai_avatar_direction == 'RIGHT':
        ai_avatar_x += ai_avatar_speed

    # Add current position to AI trail
    ai_trails.append((ai_avatar_x, ai_avatar_y))

# Draw the avatars and trails
def draw_avatars():
    pygame.draw.rect(game_display, black, [player_avatar_x, player_avatar_y, avatar_size, avatar_size])
    pygame.draw.rect(game_display, black, [ai_avatar_x, ai_avatar_y, avatar_size, avatar_size])

    for trail in player_trails:
        pygame.draw.rect(game_display, blue, [trail[0], trail[1], avatar_size, avatar_size])

    for trail in ai_trails:
        pygame.draw.rect(game_display, orange, [trail[0], trail[1], avatar_size, avatar_size])

# Game Loop
def game_loop():
    global player_avatar_x, player_avatar_y, player_avatar_x_change, player_avatar_y_change, player_trails, ai_trails

    replay = True

    while replay:
        game_over = False

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    replay = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and player_avatar_x_change != avatar_size and player_avatar_x_change != -avatar_size:
                        player_avatar_x_change = -avatar_size
                        player_avatar_y_change = 0
                    elif event.key == pygame.K_RIGHT and player_avatar_x_change != -avatar_size and player_avatar_x_change != avatar_size:
                        player_avatar_x_change = avatar_size
                        player_avatar_y_change = 0
                    elif event.key == pygame.K_UP and player_avatar_y_change != avatar_size and player_avatar_y_change != -avatar_size:
                        player_avatar_y_change = -avatar_size
                        player_avatar_x_change = 0
                    elif event.key == pygame.K_DOWN and player_avatar_y_change != -avatar_size and player_avatar_y_change != avatar_size:
                        player_avatar_y_change = avatar_size
                        player_avatar_x_change = 0

            update_avatars()

            if player_avatar_x >= width - box_margin or player_avatar_x < box_margin or player_avatar_y >= height - box_margin or player_avatar_y < box_margin:
                game_over = True

            # Check collision with trails
            if (player_avatar_x, player_avatar_y) in player_trails or (player_avatar_x, player_avatar_y) in ai_trails:
                game_over = True

            game_display.fill(white)

            draw_avatars()

            pygame.display.update()

            clock.tick(player_avatar_speed)

        # Replay prompt
        if replay:
            replay_prompt = True
            while replay_prompt:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            replay_prompt = False
                        elif event.key == pygame.K_ESCAPE:
                            replay_prompt = False
                            replay = False

        # Reset avatar positions and trails
        player_avatar_x = width / 4
        player_avatar_y = height / 2
        ai_avatar_x = 3 * width / 4
        ai_avatar_y = height / 2
        player_avatar_x_change = 0
        player_avatar_y_change = 0
        player_trails = []
        ai_trails = []

game_loop()
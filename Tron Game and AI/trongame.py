import pygame
import random

# Initialize Pygame
pygame.init()

# define ai1_last_direction
ai1_last_direction = 'UP'
ai2_last_direction = 'DOWN'

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
ai2_avatar_speed = 15
ai1_avatar_speed = 10

# Initial avatar positions
ai2_avatar_x = width / 4
ai2_avatar_y = height / 2
ai1_avatar_x = 3 * width / 4
ai1_avatar_y = height / 2

ai2_avatar_x_change = 0
ai2_avatar_y_change = 0

# Box properties
box_margin = 50
box_color = red

# Clock
clock = pygame.time.Clock()

# Trails
ai2_trails = []
ai1_trails = []

def check_collision():
    global ai2_avatar_x, ai2_avatar_y, ai2_trails

    for i in range(len(ai2_trails)):
        trail = ai2_trails[i]
        if ai2_avatar_x == trail[0] and ai2_avatar_y == trail[1]:
            if i < len(ai2_trails) - 3:
                ai2_trails = ai2_trails[:i+1]
            else:
                ai2_trails = ai2_trails[:i]
            break

def update_ai_avatar(ai_x, ai_y, ai_speed, last_direction, trails):
    global ai2_avatar_x, ai2_avatar_y, ai1_avatar_x, ai1_avatar_y, ai2_trails, ai1_trails, ai1_last_direction

    # ai2 avatar update
    ai_x += ai2_avatar_x_change
    ai_y += ai2_avatar_y_change

    # Add current position to ai2 trail
    trails.append((ai_x, ai_y))

    # AI avatar update
    # Remove the reverse of the current direction from choices
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    if last_direction == 'UP':
        directions.remove('DOWN')
    elif last_direction == 'DOWN':
        directions.remove('UP')
    elif last_direction == 'LEFT':
        directions.remove('RIGHT')
    elif last_direction == 'RIGHT':
        directions.remove('LEFT')

    ai_avatar_direction = random.choice(directions)
    last_direction = ai_avatar_direction

    if ai_avatar_direction == 'UP':
        ai_y -= ai_speed
    elif ai_avatar_direction == 'DOWN':
        ai_y += ai_speed
    elif ai_avatar_direction == 'LEFT':
        ai_x -= ai_speed
    elif ai_avatar_direction == 'RIGHT':
        ai_x += ai_speed

    # Add current position to AI trail
    trails.append((ai_x, ai_y))

    return ai_x, ai_y, last_direction, trails

# Draw the avatars and trails
def draw_avatars():
    pygame.draw.rect(game_display, black, [ai2_avatar_x, ai2_avatar_y, avatar_size, avatar_size])
    pygame.draw.rect(game_display, black, [ai1_avatar_x, ai1_avatar_y, avatar_size, avatar_size])

    for trail in ai2_trails:
        pygame.draw.rect(game_display, blue, [trail[0], trail[1], avatar_size, avatar_size])

    for trail in ai1_trails:
        pygame.draw.rect(game_display, orange, [trail[0], trail[1], avatar_size, avatar_size])

def is_collision(ai_x, ai_y, trails):
    # Check if the current position intersects with any trail segments
    for trail in trails:
        if ai_x == trail[0] and ai_y == trail[1]:
            return True
    return False

# Game Loop
def game_loop():
    global ai1_avatar_x, ai1_avatar_y, ai1_last_direction, ai1_trails
    global ai2_avatar_x, ai2_avatar_y, ai2_last_direction, ai2_trails
    global ai2_avatar_x_change, ai2_avatar_y_change  # Add global variables here

    replay = True

    while replay:
        game_over = False

        while not game_over:
            if is_collision(ai1_avatar_x, ai1_avatar_y, ai1_trails) or is_collision(ai1_avatar_x, ai1_avatar_y, ai2_trails):
                game_over = True
            if is_collision(ai2_avatar_x, ai2_avatar_y, ai2_trails) or is_collision(ai2_avatar_x, ai2_avatar_y, ai1_trails):
                game_over = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    replay = False

            # Update AI avatars
            ai1_avatar_x, ai1_avatar_y, ai1_last_direction, ai1_trails = update_ai_avatar(ai1_avatar_x, ai1_avatar_y, ai1_avatar_speed, ai1_last_direction, ai1_trails)
            ai2_avatar_x, ai2_avatar_y, ai2_last_direction, ai2_trails = update_ai_avatar(ai2_avatar_x, ai2_avatar_y, ai2_avatar_speed, ai2_last_direction, ai2_trails)

            game_display.fill(white)
            draw_avatars()
            pygame.display.update()

            # ... [Other game logic]

            clock.tick(60)  # Control the frame rate

        ai1_avatar_x, ai1_avatar_y = 3 * width / 4, height / 2
        ai2_avatar_x, ai2_avatar_y = width / 4, height / 2
        ai1_trails.clear()
        ai2_trails.clear()
        ai1_last_direction = 'UP'
        ai2_last_direction = 'DOWN'

game_loop()
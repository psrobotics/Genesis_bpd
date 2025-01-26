import pygame

import lcm
from lcm_t.exlcm import twist_t
import time

import numpy as np
import threading

lc = lcm.LCM()


# Global 3D array and lock
vel_global = np.array((0.0, 0.0, 0.0))  # Example 3D array
lock = threading.Lock()

# Function to run in the thread
def send_data():
    interval = 1 / 22 # 22hz
    x_v = 0
    y_v = 0
    ang_v = 0
    while True:
        with lock:  # Ensure thread-safe read
            x_v = vel_global[0]
            y_v = vel_global[1]
            ang_v = vel_global[2]
        
        # Send lcm twist command
        msg = twist_t()
        msg.x_vel[0] = x_v
        msg.y_vel[0] = y_v
        msg.omega_vel[0] = ang_v
        lc.publish("TWIST_T", msg.encode())
        print('lcm sent\n')

        time.sleep(interval)

# Start the thread
thread = threading.Thread(target=send_data, daemon=True)
thread.start()

# Initialize Pygame
pygame.init()

# Screen settings
width, height = 400, 250
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("keyboard ctr")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Bar properties
bar_width = 50
bar_spacing = 20
base_y = height - 50

# Velocities
x_velocity = 0.0
y_velocity = 0.0
angular_velocity = 0.0

# Velocity magnitude
linear_vv = 0.6
angular_vv = 1.3

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Main loop
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Exit condition
            running = False

    # Get pressed keys
    keys = pygame.key.get_pressed()

    # Reset velocities
    x_velocity = 0.0
    y_velocity = 0.0
    angular_velocity = 0.0

    # Update velocities based on key presses
    if keys[pygame.K_w]:  # Forward (+Y direction)
        y_velocity = linear_vv
    if keys[pygame.K_s]:  # Backward (-Y direction)
        y_velocity = -linear_vv
    if keys[pygame.K_a]:  # Left (-X direction)
        x_velocity = linear_vv
    if keys[pygame.K_d]:  # Right (+X direction)
        x_velocity = -linear_vv
    if keys[pygame.K_q]:  # Rotate counterclockwise
        angular_velocity = angular_vv
    if keys[pygame.K_e]:  # Rotate clockwise
        angular_velocity = -angular_vv

    with lock:  # Ensure thread-safe read
        vel_global[0] = y_velocity
        vel_global[1] = x_velocity
        vel_global[2] = angular_velocity

    # Clear the screen
    screen.fill(white)

    # Calculate bar heights
    x_bar_height = int(abs(x_velocity) * 200)  # Scale height
    y_bar_height = int(abs(y_velocity) * 200)
    angular_bar_height = int(abs(angular_velocity) * 200)

    # Draw the bars
    pygame.draw.rect(screen, black, (50, base_y - x_bar_height, bar_width, x_bar_height))
    pygame.draw.rect(screen, black, (150, base_y - y_bar_height, bar_width, y_bar_height))
    pygame.draw.rect(screen, black, (250, base_y - angular_bar_height, bar_width, angular_bar_height))

    # Add labels
    font = pygame.font.Font(None, 24)
    x_label = font.render("X Vel", True, black)
    y_label = font.render("Y Vel", True, black)
    angular_label = font.render("Ang Vel", True, black)

    screen.blit(x_label, (50, base_y + 10))
    screen.blit(y_label, (150, base_y + 10))
    screen.blit(angular_label, (250, base_y + 10))

    # Update the display
    pygame.display.flip()

    print(f"X v: {x_velocity:.2f}, Y v: {y_velocity:.2f}, Angular v: {angular_velocity:.2f}")

    # Control frame rate
    clock.tick(30)

# Quit Pygame
pygame.quit()

import pygame
import math

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255, 255)

class Car:
    def __init__(self):
        # Load and scale the car sprite
        self.sprite = pygame.image.load('car.png').convert_alpha()  # Use convert_alpha for transparency
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))

        self.rotated_sprite = self.sprite
        self.rotated_position = [0, 0]  # Will be updated later

        self.position = [830, 920]  # Initial top-left position
        self.angle = 0
        self.speed = 20

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = []
        self.drawing_radars = []
        self.alive = True
        self.distance = 0
        self.time = 0

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.rotated_position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Move position
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        # Keep within bounds
        self.position[0] = max(20, min(WIDTH - 120, self.position[0]))
        self.position[1] = max(20, min(HEIGHT - 120, self.position[1]))

        self.distance += self.speed
        self.time += 1

        # Update center position
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Update corners for collision detection
        length = 0.5 * CAR_SIZE_X
        angles = [30, 150, 210, 330]
        self.corners = [
            [self.center[0] + math.cos(math.radians(360 - (self.angle + a))) * length,
             self.center[1] + math.sin(math.radians(360 - (self.angle + a))) * length]
            for a in angles
        ]

        # Check collision and sensors
        self.check_collision(game_map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

        # Rotate sprite
        self.rotated_sprite, self.rotated_position = self.rotate_center(self.sprite, self.angle)

    def get_data(self):
        return [min(r[1] / 30, 10) for r in self.radars]

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        """Rotate an image and return both rotated surface and top-left to draw."""
        rotated_image = pygame.transform.rotate(image, angle)
        rect = rotated_image.get_rect(center=(self.position[0] + CAR_SIZE_X / 2,
                                              self.position[1] + CAR_SIZE_Y / 2))
        return rotated_image, rect.topleft

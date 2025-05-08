# env.py
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from car import Car, WIDTH, HEIGHT

class CarRacingEnv(gym.Env):
  metadata = {"render_modes": ["human"], "render_fps": 60}

  def __init__(self, map_file="map.png"):
    super(CarRacingEnv, self).__init__()

    self.map_file = map_file
    self.observation_space = spaces.Box(low=0, high=10, shape=(5,), dtype=np.float32)
    self.action_space = spaces.Discrete(4)
    self.car = None
    self.screen = None
    self.clock = None
    self.game_map = None
    self.render_mode = "human"

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    if self.screen is None:
      pygame.init()
      self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
      self.clock = pygame.time.Clock()
      self.game_map = pygame.image.load(f"maps/{self.map_file}").convert()

    self.car = Car()
    self.car.update(self.game_map)  # << necessary to populate radar data before get_data()

    observation = np.array(self.car.get_data(), dtype=np.float32)
    return observation, {}

  def step(self, action):
    if action == 0:
      self.car.angle += 10
    elif action == 1:
      self.car.angle -= 10
    elif action == 2:
      self.car.speed = max(12, self.car.speed - 2)
    elif action == 3:
      self.car.speed += 2

    self.car.update(self.game_map)

    observation = np.array(self.car.get_data(), dtype=np.float32)
    reward = self.car.get_reward()
    terminated = not self.car.is_alive()
    truncated = False  # You can later add a timer limit to trigger truncation
    info = {}

    return observation, reward, terminated, truncated, info

  def render(self):
    if self.render_mode != "human":
      return
    self.screen.blit(self.game_map, (0, 0))
    if self.car.is_alive():
      self.car.draw(self.screen)
    pygame.display.flip()
    self.clock.tick(60)

  def close(self):
    if self.screen:
      pygame.quit()
      self.screen = None

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from car import Car, WIDTH, HEIGHT

class CarRacingRaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, map_file="map4.png", neat_net=None):
        super().__init__()
        self.map_file = map_file
        self.neat_net = neat_net  # Injected separately
        self.render_mode = "human"

        self.observation_space = spaces.Box(low=0, high=10, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.car_ppo = None
        self.car_neat = None
        self.screen = None
        self.clock = None
        self.game_map = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
            self.clock = pygame.time.Clock()
            self.game_map = pygame.image.load(f"maps/{self.map_file}").convert()

        self.car_ppo = Car()
        self.car_neat = Car()
        self.car_neat.position[0] += 100  # Slight offset

        self.car_ppo.update(self.game_map)
        self.car_neat.update(self.game_map)

        return np.array(self.car_ppo.get_data(), dtype=np.float32), {}

    def step(self, action_ppo):
        # PPO agent action
        self._apply_action(self.car_ppo, action_ppo)

        # NEAT agent decision + action
        if self.neat_net and self.car_neat.is_alive():
            obs_neat = self.car_neat.get_data()
            action_neat = self.neat_net.activate(obs_neat)
            self._apply_action(self.car_neat, np.argmax(action_neat))

        # Update both cars
        self.car_ppo.update(self.game_map)
        self.car_neat.update(self.game_map)

        obs = np.array(self.car_ppo.get_data(), dtype=np.float32)
        reward = self.car_ppo.get_reward()
        terminated = not self.car_ppo.is_alive()
        truncated = False

        info = {
            "ppo_reward": self.car_ppo.get_reward(),
            "neat_reward": self.car_neat.get_reward(),
            "ppo_alive": self.car_ppo.is_alive(),
            "neat_alive": self.car_neat.is_alive()
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.blit(self.game_map, (0, 0))

        if self.car_ppo.is_alive():
            self.car_ppo.draw(self.screen)
        if self.car_neat.is_alive():
            self.car_neat.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _apply_action(self, car, action):
        if action == 0:
            car.angle += 10
        elif action == 1:
            car.angle -= 10
        elif action == 2:
            car.speed = max(12, car.speed - 2)
        elif action == 3:
            car.speed += 2

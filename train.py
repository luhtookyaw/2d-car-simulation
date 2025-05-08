# train.py
import os
import sys
import pygame
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env import CarRacingEnv

MODEL_FOLDER = "models"

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=100, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
        return True

def main(args):
    env = CarRacingEnv(map_file=args.map_file)
    callback = RenderCallback(env, render_freq=args.render_freq)

    # Load or create PPO model
    try:
        model = PPO.load(os.path.join(MODEL_FOLDER, args.model_name))
        model.set_env(env)
        print(f"Loaded model: {args.model_name}.zip")
    except FileNotFoundError:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
        print(f"Created new model: {args.model_name}")

    # Train
    model.learn(
        total_timesteps=args.timesteps, 
        callback=callback,
        tb_log_name=args.tb_log_name
    )

    # Save
    save_name = f"{args.model_name}_{args.map_file.split('.')[0]}"
    model.save(os.path.join(MODEL_FOLDER, save_name))
    print(f"Model saved as: {save_name}.zip")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on 2D car racing map.")
    parser.add_argument("--map_file", type=str, default="map.png", help="Path to the map file (e.g., map2.png)")
    parser.add_argument("--model_name", type=str, default="ppo_car_racer", help="Base name of the PPO model")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total timesteps to train")
    parser.add_argument("--render_freq", type=int, default=1, help="Frequency (in steps) to render environment")
    parser.add_argument("--tensorboard_log", type=str, default="ppo_logs", help="Directory for TensorBoard logs")
    parser.add_argument("--tb_log_name", type=str, default="PPO_CarRacer", help="Run name for TensorBoard")

    args = parser.parse_args()
    main(args)

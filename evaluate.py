import os
import sys
import time
import pygame
import argparse
from stable_baselines3 import PPO
from env import CarRacingEnv

def evaluate(model_path, map_file, num_episodes=5, render=True, delay=0.03):
    env = CarRacingEnv(map_file=map_file)
    model = PPO.load(os.path.join("models", model_path), env=env)

    print("Learning rate:", model.learning_rate)
    print("Gamma (discount):", model.gamma)
    print("Clip range:", model.clip_range)
    print("GAE lambda:", model.gae_lambda)
    print("Value loss coefficient:", model.vf_coef)
    print("Entropy loss coefficient:", model.ent_coef)
    print("Batch size:", model.batch_size)
    print("N steps:", model.n_steps)
    print("Number of epochs:", model.n_epochs)
    print("Policy:", model.policy)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if render:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit(0)

        print(f"🏁 Episode {ep+1} | Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PPO agent on a car racing map.")
    parser.add_argument("--model", type=str, default="ppo_car_racer_final", help="Name/path of the PPO model (no .zip needed)")
    parser.add_argument("--map_file", type=str, default="map4.png", help="Map file to evaluate on")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering for faster evaluation")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between frames when rendering")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        map_file=args.map_file,
        num_episodes=args.episodes,
        render=not args.no_render,
        delay=args.delay
    )

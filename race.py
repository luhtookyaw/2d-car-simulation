import sys
import pickle
import pygame
from stable_baselines3 import PPO
from race_env import CarRacingRaceEnv

# Load NEAT model
with open("models/neat_winner.pkl", "rb") as f:
    neat_genome = pickle.load(f)

# Load NEAT network
import neat
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config.txt"
)
neat_net = neat.nn.FeedForwardNetwork.create(neat_genome, config)

# Create race env
env = CarRacingRaceEnv(map_file="map4.png", neat_net=neat_net)

# Load PPO model
ppo_model = PPO.load("models/ppo_car_racer_map4", env=env)

ppo_wins = 0
neat_wins = 0
draws = 0
episodes = 5

for ep in range(1, episodes + 1):
    obs, _ = env.reset()
    done = False
    ppo_total_reward = 0
    neat_total_reward = 0

    while not done:
        action, _ = ppo_model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        ppo_total_reward += info["ppo_reward"]
        neat_total_reward += info["neat_reward"]

    print(f"\n🏁 Episode {ep}:")
    print(f"NEAT reward: {neat_total_reward:.2f}")
    print(f"PPO reward: {ppo_total_reward:.2f}")

    if neat_total_reward > ppo_total_reward:
        print("✅ NEAT wins this round!")
        neat_wins += 1
    elif ppo_total_reward > neat_total_reward:
        print("✅ PPO wins this round!")
        ppo_wins += 1
    else:
        print("⚖️ It's a draw!")
        draws += 1

env.close()

print("\n=== Final Results ===")
print(f"🏆 NEAT wins: {neat_wins}")
print(f"🏆 PPO wins: {ppo_wins}")
print(f"🤝 Draws: {draws}")

if neat_wins > ppo_wins:
    print("🎉 NEAT is the overall winner!")
elif ppo_wins > neat_wins:
    print("🎉 PPO is the overall winner!")
else:
    print("🤝 It's an overall draw!")


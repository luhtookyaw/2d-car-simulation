import sys
import pickle
import pygame
from stable_baselines3 import PPO
from race_env import CarRacingRaceEnv
import neat

# Load NEAT model
with open("models/neat_winner_final.pkl", "rb") as f:
    neat_genome = pickle.load(f)

# Load NEAT network
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config.txt"
)
neat_net = neat.nn.FeedForwardNetwork.create(neat_genome, config)

# Create evaluation environment
env = CarRacingRaceEnv(map_file="map4.png", neat_net=neat_net)

# Load PPO model
ppo_model = PPO.load("models/ppo_car_racer_final", env=env)

# Evaluation parameters
episodes = 10
ppo_wins = 0
neat_wins = 0
draws = 0

ppo_rewards = []
neat_rewards = []
episode_lengths = []

# Run evaluation episodes
for ep in range(1, episodes + 1):
    obs, _ = env.reset()
    done = False
    ppo_total_reward = 0
    neat_total_reward = 0
    step_count = 0

    while not done:
        action, _ = ppo_model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        ppo_total_reward += info["ppo_reward"]
        neat_total_reward += info["neat_reward"]
        step_count += 1

    # Logging results
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

    # Store for averages
    neat_rewards.append(neat_total_reward)
    ppo_rewards.append(ppo_total_reward)
    episode_lengths.append(step_count)

env.close()

# Print final results
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

# Averages
avg_neat_reward = sum(neat_rewards) / episodes
avg_ppo_reward = sum(ppo_rewards) / episodes
avg_ep_len = sum(episode_lengths) / episodes

print(f"\n=== Averages over {episodes} episodes ===")
print(f"📈 Avg NEAT reward: {avg_neat_reward:.2f}")
print(f"📈 Avg PPO reward: {avg_ppo_reward:.2f}")
print(f"⏱️ Avg episode length: {avg_ep_len:.2f} steps")

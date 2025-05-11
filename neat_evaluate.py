import sys
import pickle
import neat
import pygame
from neat_car import Car

WIDTH = 1920
HEIGHT = 1080

def evaluate_trained_model(net, episodes=5):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    game_map = pygame.image.load('maps/map4.png').convert()

    total_fitness = 0

    for ep in range(episodes):
        car = Car()
        clock = pygame.time.Clock()

        while car.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            inputs = car.get_data()
            output = net.activate(inputs)
            choice = output.index(max(output))

            # Control logic
            if choice == 0:
                car.angle += 10
            elif choice == 1:
                car.angle -= 10
            elif choice == 2 and car.speed - 2 >= 12:
                car.speed -= 2
            else:
                car.speed += 2

            car.update(game_map)

            screen.blit(game_map, (0, 0))
            car.draw(screen)

            pygame.display.flip()
            clock.tick(60)

        ep_fitness = car.get_reward()
        total_fitness += ep_fitness
        print(f"Episode {ep + 1} fitness: {ep_fitness:.2f}")

    avg_fitness = total_fitness / episodes
    print(f"\nAverage fitness over {episodes} episodes: {avg_fitness:.2f}")
    pygame.quit()

if __name__ == "__main__":
    # Load NEAT config
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config.txt"
    )

    # Load the saved genome
    with open("models/neat_winner_final.pkl", "rb") as f:
        winner = pickle.load(f)

    # Rebuild the trained neural network
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    evaluate_trained_model(net, episodes=10)

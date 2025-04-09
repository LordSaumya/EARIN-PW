import random
import math
import matplotlib.pyplot as plt
import numpy as np # Import numpy for linspace


# Constants
X_BOUNDS = (-15, 5)
Y_BOUNDS = (-3, 3)
GLOBAL_OPTIMUM_POS = (-10, 1)
GLOBAL_OPTIMUM_VAL = 0.0

# Seed for reproducibility (tuned)
random.seed(42)

# --- Bukin Function ---
def bukin(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

# --- Fitness Function ---
def fitness_function(x, y):
    # Negative Bukin function for minimisation
    return -bukin(x, y)

class Genotype:
    def __init__(self, x, y):
        # Ensure initial values are within bounds
        self.x = max(X_BOUNDS[0], min(X_BOUNDS[1], x))
        self.y = max(Y_BOUNDS[0], min(Y_BOUNDS[1], y))
        # Calculate fitness based on clamped values
        self.fitness = fitness_function(self.x, self.y)

    def __str__(self):
        return f"Genotype(x={self.x:.4f}, y={self.y:.4f}, fitness={self.fitness:.6f}, bukin={-self.fitness:.6f})"

    def mutate(self, mutation_rate, mutation_strength):
        if random.random() < mutation_rate:
            # Apply Gaussian noise
            self.x += random.gauss(0, mutation_strength)
            self.y += random.gauss(0, mutation_strength)
            self.x = max(X_BOUNDS[0], min(X_BOUNDS[1], self.x))
            self.y = max(Y_BOUNDS[0], min(Y_BOUNDS[1], self.y))

            # Recalculate fitness
            self.fitness = fitness_function(self.x, self.y)

    def crossover(self, other):
        # Random interpolation using alpha
        alpha = random.random()
        child_x = alpha * self.x + (1 - alpha) * other.x
        child_y = alpha * self.y + (1 - alpha) * other.y

        return Genotype(child_x, child_y)

class Population:
    def __init__(self, size):
        self.size = size
        self.genotypes = [
            Genotype(random.uniform(X_BOUNDS[0], X_BOUNDS[1]),
                     random.uniform(Y_BOUNDS[0], Y_BOUNDS[1]))
            for _ in range(size)
        ]

        self.update_best_genotype()

    def update_best_genotype(self):
        self.best_genotype = max(self.genotypes, key=lambda g: g.fitness)

    def tournament_selection(self, tournament_size):
        actual_tournament_size = min(tournament_size, len(self.genotypes))
        selected_contenders = random.sample(self.genotypes, actual_tournament_size)

        # Return contender with max fitness
        return max(selected_contenders, key=lambda g: g.fitness)

    def evolve(self, mutation_rate, mutation_strength, tournament_size, elitism_ratio=0.1):
        new_genotypes = []
        num_elites = int(self.size * elitism_ratio)

        # Keep the best genotypes in the population
        self.genotypes.sort(key=lambda g: g.fitness, reverse=True)
        new_genotypes.extend(self.genotypes[:num_elites])

        # Generate offspring
        num_offspring = self.size - num_elites
        for _ in range(num_offspring):
            # Selection
            parent1 = self.tournament_selection(tournament_size)
            parent2 = self.tournament_selection(tournament_size)

            # Crossover
            child = parent1.crossover(parent2)

            # Mutation
            child.mutate(mutation_rate, mutation_strength)

            new_genotypes.append(child)

        # Replace old population and update best genotype
        self.genotypes = new_genotypes
        self.update_best_genotype()

def main():
    # Parameters
    generations = 2000
    population_size = 500   # tuned
    mutation_rate = 0.5     # Probability of mutation
    mutation_strength = 1.00251   # Standard deviation for Gaussian mutation (tuned)
    tournament_size = 10    # Number of individuals in tournament selection (tuned)
    elitism_ratio = 0.2     # Percentage of top individuals to keep (tuned)

    population = Population(population_size)
    best_fitness_history = [-population.best_genotype.fitness] # Store Bukin values of best genotype for plotting

    for generation in range(generations):
        population.evolve(mutation_rate, mutation_strength, tournament_size, elitism_ratio)
        current_best_bukin = -population.best_genotype.fitness
        best_fitness_history.append(current_best_bukin)

        if (generation + 1) % 20 == 0: # Print progress periodically
             print(f"Generation {generation + 1:4d}: Best {population.best_genotype}")

    # Results
    print("\n--- Optimisation Finished ---")
    print(f"Final best genotype: {population.best_genotype}")
    print(f"Found minimum value: {-population.best_genotype.fitness:.6f}")
    print(f"Known global minimum: {GLOBAL_OPTIMUM_VAL:.6f} at {GLOBAL_OPTIMUM_POS}")
    distance = math.sqrt((population.best_genotype.x - GLOBAL_OPTIMUM_POS[0])**2 +
                         (population.best_genotype.y - GLOBAL_OPTIMUM_POS[1])**2)
    print(f"Distance to global optimum: {distance:.6f}")

    # Plots

    # Plot 1: Best Bukin value over generations
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history)
    plt.title('Best Bukin Value Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Bukin Function Value')
    plt.axhline(y=GLOBAL_OPTIMUM_VAL, color='r', linestyle='--', label=f'Global Minimum ({GLOBAL_OPTIMUM_VAL})')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=min(0, min(best_fitness_history) - 1)) # Adjust y-axis lower limit
    plt.show()

    # Plot 2: 3D surface plot of the Bukin function with best genotype and global optimum
    x = np.linspace(X_BOUNDS[0], X_BOUNDS[1], 1000)
    y = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], 1000)
    X, Y = np.meshgrid(x, y)
    Z = bukin(X, Y)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.scatter(population.best_genotype.x, population.best_genotype.y, -population.best_genotype.fitness, color='red', s=100, label='Best Genotype')
    ax.scatter(GLOBAL_OPTIMUM_POS[0], GLOBAL_OPTIMUM_POS[1], GLOBAL_OPTIMUM_VAL, color='blue', s=100, label='Global Optimum')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Bukin Function Value')
    ax.set_title('Bukin Function Surface with Best Genotype and Global Optimum')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

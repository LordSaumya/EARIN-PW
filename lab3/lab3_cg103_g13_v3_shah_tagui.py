import random
import math
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from functools import partial

# Constants
X_BOUNDS = (-15, 5)
Y_BOUNDS = (-3, 3)
GLOBAL_OPTIMUM_POS = (-10, 1)
GLOBAL_OPTIMUM_VAL = 0.0

# Seed for reproducibility (tuned)
# random.seed(2317)

# Bukin Function
def bukin(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

# Fitness function (negative Bukin function for minimisation)
def fitness_function(x, y):
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

    def evolve(self, mutation_rate, mutation_strength, tournament_size, elitism_ratio=0.1, crossover_rate=0.8):
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

            # Crossover with probability crossover_rate, otherwise clone a parent
            if random.random() < crossover_rate:
                child = parent1.crossover(parent2)
            else:
                # No crossover, just clone one of the parents (randomly choose which one)
                child = Genotype(parent1.x, parent1.y) if random.random() < 0.5 else Genotype(parent2.x, parent2.y)
            
            # Mutation
            child.mutate(mutation_rate, mutation_strength)

            new_genotypes.append(child)

        # Replace old population and update best genotype
        self.genotypes = new_genotypes
        self.update_best_genotype()

def run_trial(params, tuning_generations):    
    print(f"\nTrial {params['trial_id']+1} with parameters:")
    for key, value in params.items():
        if key != 'trial_id':
            print(f"  {key}: {value}")
    
    # Run a shorter version of main with these parameters
    fitness = main(
        generations=tuning_generations,
        population_size=params['population_size'],
        mutation_rate=params['mutation_rate'],
        mutation_strength=params['mutation_strength'],
        tournament_size=params['tournament_size'],
        elitism_ratio=params['elitism_ratio'],
        crossover_rate=params['crossover_rate'],
        verbose=False  # Don't print generation progress during tuning
    )
    
    print(f"Result - Bukin function value: {fitness:.6f}")
    
    return {
        'params': {k: v for k, v in params.items() if k != 'trial_id'},
        'fitness': fitness,
        'trial_id': params['trial_id']
    }

def tune_hyperparameters(num_trials, tuning_generations):
    print("\n--- Hyperparameter Tuning ---")
    
    # Define parameter ranges to search
    param_grid = {
        'population_size': np.arange(100, 1501, 100).tolist(),  # 100 to 1500 in steps of 100
        'mutation_rate': np.arange(0.1, 1.05, 0.05).tolist(),  # 0.1 to 1.0 in steps of 0.05
        'mutation_strength': np.arange(0.1, 2.05, 0.05).tolist(),  # 1.0 to 2.0 in steps of 0.05
        'tournament_size': [5, 10, 20, 30],  # Fixed sizes for simplicity
        'elitism_ratio': np.arange(0.0, 1.1, 0.1).tolist(),  # 0.0 to 1.0 in steps of 0.1
        'crossover_rate': np.arange(0.5, 1.01, 0.05).tolist(),  # 0.5 to 1.0 in steps of 0.05
    }
    
    # Generate parameter combinations for all trials
    all_params = []
    for trial_id in range(num_trials):
        # Randomly select parameters from the grid
        params = {
            'population_size': random.choice(param_grid['population_size']),
            'mutation_rate': random.choice(param_grid['mutation_rate']),
            'mutation_strength': random.choice(param_grid['mutation_strength']),
            'tournament_size': random.choice(param_grid['tournament_size']),
            'elitism_ratio': random.choice(param_grid['elitism_ratio']),
            'crossover_rate': random.choice(param_grid['crossover_rate']),
            'trial_id': trial_id,
        }
        all_params.append(params)
    
    # Determine the number of processes to use (cpu_count - 1 to leave one CPU free)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running {num_trials} trials using {num_processes} parallel processes")
    
    # Create a partial function with the fixed tuning_generations parameter
    run_trial_fixed = partial(run_trial, tuning_generations=tuning_generations)
    
    # Run trials in parallel using multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(run_trial_fixed, all_params)
    
    # Find the best parameters
    best_result = min(results, key=lambda x: x['fitness'])
    best_params = best_result['params']
    best_fitness = best_result['fitness']
    
    print("\n--- Hyperparameter Tuning Complete ---")
    print("Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Bukin function value: {best_fitness:.6f}")
    
    return best_params

def main(generations, population_size, mutation_rate, mutation_strength, tournament_size, elitism_ratio, crossover_rate=0.8, verbose=True):
    population = Population(population_size)
    
    if verbose:
        print(f"Initial best genotype: {population.best_genotype}")
    
    best_fitness_history = [-population.best_genotype.fitness] # Store Bukin values of best genotype for plotting

    for generation in range(generations):
        population.evolve(mutation_rate, mutation_strength, tournament_size, elitism_ratio, crossover_rate)
        current_best_bukin = -population.best_genotype.fitness
        best_fitness_history.append(current_best_bukin)

        if verbose and (generation + 1) % 20 == 0: # Print progress periodically
            print(f"Generation {generation + 1:4d}: Best {population.best_genotype}")
    
    # Results
    if verbose: # Do not print and plot if tuning
        print("\n--- Optimisation Finished ---")
        print(f"Final best genotype: {population.best_genotype}")
        print(f"Found minimum value: {-population.best_genotype.fitness:.6f}")
        print(f"Known global minimum: {GLOBAL_OPTIMUM_VAL:.6f} at {GLOBAL_OPTIMUM_POS}")

        # # Plots # Uncomment to enable plotting

        # # Plot 1: Best Bukin value over generations
        # plt.figure(figsize=(10, 6))
        # plt.plot(best_fitness_history)
        # plt.title('Best Bukin Value Over Generations')
        # plt.xlabel('Generation')
        # plt.ylabel('Best Bukin Function Value')
        # plt.axhline(y=GLOBAL_OPTIMUM_VAL, color='r', linestyle='--', label=f'Global Minimum ({GLOBAL_OPTIMUM_VAL})')
        # plt.grid(True)
        # plt.legend()
        # plt.ylim(bottom=min(0, min(best_fitness_history) - 1)) # Adjust y-axis lower limit
        # plt.show()

        # # Plot 2: 3D surface plot of the Bukin function with best genotype and global optimum
        # x = np.linspace(X_BOUNDS[0], X_BOUNDS[1], 1000)
        # y = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], 1000)
        # X, Y = np.meshgrid(x, y)
        # Z = bukin(X, Y)
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        # ax.scatter(population.best_genotype.x, population.best_genotype.y, -population.best_genotype.fitness, color='red', s=100, label='Best Genotype')
        # ax.scatter(GLOBAL_OPTIMUM_POS[0], GLOBAL_OPTIMUM_POS[1], GLOBAL_OPTIMUM_VAL, color='blue', s=100, label='Global Optimum')
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Bukin Function Value')
        # ax.set_title('Bukin Function Surface with Best Genotype and Global Optimum')
        # ax.legend()
        # plt.show()
    
    return -population.best_genotype.fitness, population  # Return both fitness and population

if __name__ == "__main__":
    # Tune hyperparameters first
    do_tuning = False  # Set to False to skip tuning and use predefined parameters
    
    if do_tuning:
        # Run tuning with fewer generations to find optimal parameters
        best_params = tune_hyperparameters(num_trials=3000, tuning_generations=300)
        
        # Run with the best parameters found and more generations
        main(
            generations=2000,
            population_size=best_params['population_size'],
            mutation_rate=best_params['mutation_rate'],
            mutation_strength=best_params['mutation_strength'],
            tournament_size=best_params['tournament_size'],
            elitism_ratio=best_params['elitism_ratio'],
            crossover_rate=best_params['crossover_rate']
        )
    else:
        # Use predefined parameters
        generations = 500
        population_size = 500
        mutation_rate = 0.5 * 2.0
        mutation_strength = 1.00251 * 2.0
        tournament_size = 10
        elitism_ratio = 0.2
        crossover_rate = 1.0
        
        # Get both the fitness and population from main
        best_fitness, final_population = main(
            generations=generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            tournament_size=tournament_size,
            elitism_ratio=elitism_ratio,
            crossover_rate=crossover_rate
        )
        
        # Calculate population statistics
        fitness_values = [-g.fitness for g in final_population.genotypes]  # Convert to Bukin values
        avg_fitness = np.mean(fitness_values)
        std_dev = np.std(fitness_values)
        
        print("\n--- Population Statistics ---")
        print(f"Average Bukin value: {avg_fitness:.6f}")
        print(f"Standard deviation: {std_dev:.6f}")
        print(f"Best Bukin value: {best_fitness:.6f}")

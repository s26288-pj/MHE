import random
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from open_shop_scheduling import display_schedule, generate_random_solution, simulate_job_execution, generate_jobs

warnings.simplefilter(action='ignore', category=FutureWarning)

def initialize_population(pop_size, num_machines, num_jobs):
    """ Initialize a population of random solutions.

    :param pop_size: Size of the population.
    :param num_jobs: Number of jobs to be scheduled.
    :param num_machines: Number of machines available for scheduling.
    :return: A list of random schedules.
    """
    return [generate_random_solution(num_machines, num_jobs) for _ in range(pop_size)]


def calculate_population_fitness(population, processing_times):
    """ Calculate the fitness of each individual in the population.

    :param population: List of schedules.
    :param processing_times: Matrix of processing times for jobs on machines.
    :return: A list of fitness values for the population.
    """
    fitness = []
    for schedule in population:
        total_time = simulate_job_execution(schedule, processing_times)
        fitness.append(1 / total_time)  # Invert total_time as fitness (higher fitness for smaller total time)
    return fitness


def selection(population, fitness, num_parents):
    """ Select parents for crossover based on their fitness.

    :param population: List of schedules.
    :param fitness: List of fitness values for the population.
    :param num_parents: Number of parents to select.
    :return: A list of selected parent schedules.
    """
    selected_indices = np.random.choice(len(population), size=num_parents, replace=False, p=fitness / np.sum(fitness))
    return [population[idx] for idx in selected_indices]


def crossover(parents, num_offsprings, num_points):
    """ Perform crossover to generate offspring schedules.

    :param parents: List of parent schedules.
    :param num_offsprings: Number of offspring to produce.
    :param num_points: Number of reference points for crossover. Can be 1 or 2.
    :return: A list of offspring schedules.
    """
    offsprings = []
    for _ in range(num_offsprings):
        parent1, parent2 = random.sample(parents, 2)
        if num_points == 1:
            crossover_point = random.randint(1, len(parent1["Machine_1"]))
            offspring = pd.concat((parent1.copy().iloc[:crossover_point], parent2.copy().iloc[crossover_point:]))
        else:
            point1, point2 = sorted(random.sample(range(1, len(parent1["Machine_1"]) - 1), 2))
            offspring = pd.concat((parent1.copy().iloc[:point1], parent2.copy().iloc[point1:point2], parent1.copy().iloc[point2:]))
        offsprings.append(offspring)
    return offsprings


def swap_mutation(offsprings, mutation_rate, num_machines):
    """ Apply swap mutation to offspring schedules.

    :param offsprings: List of offspring schedules.
    :param mutation_rate: Probability of mutation for each job.
    :param num_machines: Number of machines available for scheduling.
    :return: A list of mutated offspring schedules.
    """
    for offspring in offsprings:
        jobs_len = len(offspring[f"Machine_1"]) - 1
        for idx in range(1, num_machines + 1):
            if random.random() < mutation_rate:
                offspring[f"Machine_{idx}"][0], offspring[f"Machine_{idx}"][jobs_len] = \
                    offspring[f"Machine_{idx}"][jobs_len], offspring[f"Machine_{idx}"][0]
    return offsprings


def inversion_mutation(offsprings, mutation_rate, num_machines):
    """ Apply inversion mutation to offspring schedules.

    :param offsprings: List of offspring schedules.
    :param mutation_rate: Probability of mutation for each job.
    :param num_machines: Number of machines available for scheduling.
    :return: A list of mutated offspring schedules.
    """
    for offspring in offsprings:
        for idx in range(1, num_machines + 1):
            if random.random() < mutation_rate:
                start, end = sorted(random.sample(range(len(offspring[f"Machine_{idx}"])), 2))
                offspring[f"Machine_{idx}"].iloc[start:end] = offspring[f"Machine_{idx}"].iloc[start:end][::-1]
    return offsprings


def genetic_algorithm(processing_times, num_machines, num_jobs, pop_size=100, num_generations=100, num_parents=50,
                      num_offsprings=50, mutation_rate=0.1, crossover_points=1, mutation_method="swap", show_plots=True):
    """ Genetic Algorithm for solving the open-shop scheduling problem.

    :param processing_times: Matrix of processing times for jobs on machines.
    :param num_machines: Number of machines available for scheduling.
    :param num_jobs: Number of jobs to be scheduled.
    :param pop_size: Size of the population.
    :param num_generations: Number of generations to run the algorithm.
    :param num_parents: Number of parents to select for crossover.
    :param num_offsprings: Number of offspring to produce in each generation.
    :param mutation_rate: Probability of mutation for each job.
    :param crossover_points: Number of reference points for crossover.
    :param mutation_method: Method used for mutation of offsprings.
    :param show_plots: Boolean indicating whether to display plots during optimization.
    :return: The best schedule found, its total time, and a list of best times over generations.
    """
    population = initialize_population(pop_size, num_machines, num_jobs)
    best_solution = None
    best_time = float('inf')
    total_time = []

    start_time = time.time()

    for gen in range(num_generations):
        fitness = calculate_population_fitness(population, processing_times)

        if show_plots and gen % 10 == 0:
            best_idx = np.argmax(fitness)
            display_schedule(population[best_idx], processing_times)
            plt.pause(0.1)

        parents = selection(population, fitness, num_parents)
        offsprings = crossover(parents, num_offsprings, crossover_points)

        if mutation_method == 'swap':
            offsprings = swap_mutation(offsprings, mutation_rate, num_machines)
        elif mutation_method == 'inversion':
            offsprings = inversion_mutation(offsprings, mutation_rate, num_machines)
        else:
            raise ValueError("Invalid mutation method")

        population = parents + offsprings

        # Evaluate the best solution in the current generation
        current_best_idx = np.argmax(fitness)
        current_best_time = 1 / fitness[current_best_idx]  # Transform fitness back to total time
        if current_best_time < best_time:
            best_time = current_best_time
            best_solution = population[current_best_idx]

        total_time.append(best_time)

    end_time = time.time()

    if show_plots:
        plt.ioff()
        plt.show()

    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Final total time: {best_time:.4f}")

    return best_solution, best_time, total_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Genetic Algorithm for Open-shop Scheduling')
    parser.add_argument('--machines', type=int, default=5, help='Number of machines')
    parser.add_argument('--jobs', type=int, default=10, help='Number of jobs')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--num_generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--num_parents', type=int, default=50, help='Number of parents to select')
    parser.add_argument('--num_offsprings', type=int, default=50, help='Number of offsprings to produce')
    parser.add_argument('--mutation_rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--crossover_points', type=int, default=2, help='Number of reference points for crossover')
    parser.add_argument('--mutation_method', type=str, default='swap', choices=['swap', 'inversion'], help='Mutation method to use')
    parser.add_argument('--show_plots', action="store_true", help='Show plots during optimization')

    args = parser.parse_args()

    processing_times = generate_jobs(args.machines, args.jobs)

    print(f"Running Genethic Algorithm with {args.num_generations} generations, {args.num_parents} parents, \
        {args.num_offsprings} offsprings, and {args.mutation_rate} mutation rate")

    solution, total_time, total_times = genetic_algorithm(processing_times, args.machines, args.jobs,
                                                          pop_size=args.pop_size,
                                                          num_generations=args.num_generations,
                                                          num_parents=args.num_parents,
                                                          num_offsprings=args.num_offsprings,
                                                          mutation_rate=args.mutation_rate,
                                                          crossover_points=args.crossover_points,
                                                          mutation_method=args.mutation_method,
                                                          show_plots=args.show_plots)

    if not args.show_plots:
        plt.plot(total_times)
        plt.xlabel('Generation')
        plt.ylabel('Total time')
        plt.title('Genetic Algorithm Convergence')
        plt.show()

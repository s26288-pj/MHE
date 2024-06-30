import itertools
import math
import random
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_random_solution(num_machines, num_jobs):
    """ Generate random solution for Open Shop Scheduling.

    :param num_machines: number of devices that will be responsible for finishing jobs
    :param num_jobs: number of jobs to be distributed among machines
    """
    random_solution = {}
    for i in range(1, num_machines+1):
        random_solution[f"Machine_{i}"] = np.random.permutation(range(1, num_jobs + 1))
    data_frame = pd.DataFrame(random_solution)
    return data_frame


def generate_all_solutions(num_machines, num_jobs):
    """ Generate all possible solutions for Open Shop Scheduling.

    :param num_machines: number of devices that will be responsible for finishing jobs
    :param num_jobs: number of jobs to be distributed among machines
    """
    job_permutations = list(itertools.permutations(range(1, num_jobs + 1)))
    all_solutions = list(itertools.product(job_permutations, repeat=num_machines))

    solutions = []
    for solution in all_solutions:
        solution_dict = {f"Machine_{i + 1}": solution[i] for i in range(num_machines)}
        data_frame = pd.DataFrame(solution_dict)
        solutions.append(data_frame)

    return solutions


def generate_neighbours(solution, num_machines):
    """ Generate 2 neighbours randomly. Neighbour is almost an identical copy of solution table, but with 2 swapped jobs.

    :param solution: current solution generated by generate_random_solution
    :param num_machines: number of machines for Open Shop Scheduling
    """
    rand_machine_1, rand_machine_2 = random.sample(range(1, num_machines - 1), 2)
    a, b = random.sample(range(len(solution[f"Machine_{rand_machine_1}"])), 2)
    c, d = random.sample(range(len(solution[f"Machine_{rand_machine_2}"])), 2)

    neighbour1 = solution.copy()
    neighbour2 = solution.copy()
    neighbours = [neighbour1, neighbour2]

    neighbours[0][f"Machine_{rand_machine_1}"][a], neighbours[0][f"Machine_{rand_machine_1}"][b] = \
        neighbours[0][f"Machine_{rand_machine_1}"][b], neighbours[0][f"Machine_{rand_machine_1}"][a]

    neighbours[1][f"Machine_{rand_machine_2}"][c], neighbours[1][f"Machine_{rand_machine_2}"][d] = \
        neighbours[1][f"Machine_{rand_machine_2}"][d], neighbours[1][f"Machine_{rand_machine_2}"][c]

    return neighbour1, neighbour2


def generate_jobs(num_machines, num_jobs):
    """ Generates jobs with random execution times for each machine.

    :param num_machines: number of machines used for Open Shop Scheduling
    :param num_jobs: number of jobs that is distributed among devices
    """
    solution_execution_time = {}
    for i in range(1, num_machines+1):
        solution_execution_time[f"Machine_{i}"] = [random.randint(1, num_jobs+10) for i in range(num_jobs)]
    time_frame = pd.DataFrame(solution_execution_time)
    return time_frame


def simulate_job_execution(solution, processing_times):
    schedule = solution.copy()
    num_jobs = len(schedule["Machine_1"])
    current_tasks = {machine: schedule[machine][0] for machine in schedule}
    wait_time = {machine: 0 for machine in schedule}
    execution_time = {machine: 0 for machine in schedule}
    executed_tasks = {machine: 0 for machine in schedule}
    is_task_waiting = {machine: False for machine in schedule}

    def is_task_waiting_for_execution():
        for task in current_tasks.values():
            machines_with_task = [machine for machine, job in current_tasks.items() if job == task]
            first_machine = machines_with_task[0]
            for machine in machines_with_task:
                if machine == first_machine:
                    is_task_waiting[machine] = False
                else:
                    is_task_waiting[machine] = True

    is_task_waiting_for_execution()
    timer = 0

    while True:
        for machine in schedule:
            current_job = current_tasks[machine]
            if current_job != 0:
                job_time = processing_times[machine][current_job - 1]
            else:
                if executed_tasks[machine] == num_jobs:
                    continue
                else:
                    job_time = processing_times[machine][executed_tasks[machine]]

            if is_task_waiting[machine]:
                wait_time[machine] += 1

            if timer == execution_time[machine] + job_time + wait_time[machine]:
                executed_tasks[machine] += 1
                execution_time[machine] = execution_time[machine] + job_time + wait_time[machine]
                wait_time[machine] = 0
                if executed_tasks[machine] == num_jobs:
                    current_tasks[machine] = 0
                else:
                    current_tasks[machine] = schedule[machine][executed_tasks[machine]]
                is_task_waiting_for_execution()

        timer += 1

        if all(value == num_jobs for value in executed_tasks.values()):
            break

    return max(execution_time.values())


def display_schedule(total_times, title):
    """ Displays graphs that help comparing different solutions.

    :param total_times: current total time of job execution
    :param title: title of th graph to be displayed
    """
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(range(len(total_times)), total_times, label='Max Execution Time')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Execution Time')
    ax.set_title(title)
    ax.legend()
    plt.pause(0.01)
    plt.show()


def hill_climbing(initial_solution, processing_times, num_machines, iterations=1000, show_plots=True):
    current_solution = initial_solution.copy()
    current_time = simulate_job_execution(current_solution, processing_times)
    total_times = [current_time]
    start_time = time.time()

    for iteration in range(iterations):
        neighbor1, neighbor2 = generate_neighbours(current_solution, num_machines)
        neighbor_time1 = simulate_job_execution(neighbor1, processing_times)
        neighbor_time2 = simulate_job_execution(neighbor2, processing_times)
        if neighbor_time1 < current_time:
            current_solution = neighbor1
            current_time = neighbor_time1
        if neighbor_time2 < current_time:
            current_solution = neighbor2
            current_time = neighbor_time2
        total_times.append(current_time)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Final total time: {total_times[-1]:.4f}")

    if show_plots:
        display_schedule(total_times, "Hill Climbing")

    return current_solution, total_times[-1], total_times


def tabu_search(initial_solution, processing_times, num_machines, tabu_size=10, max_iterations=100, show_plots=True):
    """ Tabu Search algorithm function.

    :param initial_solution: current solution
    :param processing_times: table of processing times of the jobs for each machine
    :param num_machines: number of machines used for Open Shop Scheduling
    :param tabu_size: size of elements that will be unavailable for re-calculations
    :param max_iterations: number of maximum iterations allowed by algorithm
    :param show_plots: boolean that is responsible for turning on/off generating of the graphs
    """
    current_solution = initial_solution.copy()
    current_time = simulate_job_execution(current_solution, processing_times)
    best_solution = current_solution.copy()
    best_time = current_time

    tabu_list = []

    total_time = [current_time]
    start_time = time.time()

    for _ in range(max_iterations):
        neighbour1, neighbour2 = generate_neighbours(current_solution, num_machines)
        neighbour_time = simulate_job_execution(neighbour1, processing_times)

        if neighbour_time < current_time and not any(df.equals(neighbour1) for df in tabu_list):
            current_solution = neighbour1.copy()
            current_time = neighbour_time

            if current_time < best_time:
                best_solution = current_solution.copy()
                best_time = current_time

            # Add the current solution to the tabu list
            tabu_list.append(current_solution)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        total_time.append(current_time)

    end_time = time.time()

    if show_plots:
        display_schedule(total_time, "Tabu Search")

    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Final total time: {best_time:.4f}")

    return best_solution, best_time, total_time


def full_search(num_machines, num_jobs):
    """ Full search algorithm function.

    :param num_machines: number of machines used for Open Shop Scheduling
    :param num_jobs: number of jobs that is distributed among devices
    """
    best_time = 10000
    solutions = generate_all_solutions(num_machines, num_jobs)
    processing_times = generate_jobs(num_machines=num_machines, num_jobs=num_jobs)

    start_time = time.time()

    for solution in solutions:
        solution_time = simulate_job_execution(solution, processing_times)
        if solution_time < best_time:
            best_time = solution_time

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return best_time


def simulated_annealing(initial_solution, processing_times, num_machines, iterations, show_plots):
    current_time = simulate_job_execution(initial_solution, processing_times)
    current_solution = initial_solution
    best_time = current_time
    best_solution = current_solution

    total_time = [best_time]

    start_time = time.time()

    for iter in range(iterations):
        temp = 1 - iter / iterations
        print(iter)
        new_solution = generate_random_solution(num_machines, len(current_solution["Machine_1"]))
        new_time = simulate_job_execution(new_solution, processing_times)

        if new_time < best_time or math.exp((best_time - new_time) / temp) > np.random.rand():
            current_solution = new_solution
            current_time = new_time

            if current_time < best_time:
                best_solution = current_solution
                best_time = current_time

        total_time.append(current_time)

    end_time = time.time()

    if show_plots:
        plt.show()

    if show_plots:
        display_schedule(total_time, "Simulated Annealing")

    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Final best time: {best_time:.4f}")

    return best_solution, best_time, total_time


def compare_algorithms(num_machines, num_jobs, iterations, tabu_size, show_plots):
    """ Compare the performance of Hill Climbing, Tabu Search, and Full Search algorithms.

    :param num_jobs: number of jobs that is distributed among devices
    :param num_machines: number of machines used for Open Shop Scheduling
    :param iterations: number of iterations for Hill Climbing and Tabu Search
    :param tabu_size: size of tabu list for Tabu Search
    :param show_plots: boolean that is responsible for turning on/off generating of the graphs
    """
    processing_times = generate_jobs(num_machines=num_machines, num_jobs=num_jobs)
    initial_solution = generate_random_solution(num_machines, num_jobs)
    #
    print("Running Hill Climbing...")
    hc_solution, hc_time, hc_total_time = hill_climbing(
        initial_solution=initial_solution,
        processing_times=processing_times,
        num_machines=num_machines,
        iterations=iterations,
        show_plots=show_plots
    )

    print("Running Tabu Search...")
    ts_solution, ts_time, ts_total_time = tabu_search(
        initial_solution=initial_solution,
        processing_times=processing_times,
        num_machines=num_machines,
        tabu_size=tabu_size,
        max_iterations=iterations,
        show_plots=show_plots
    )

    print("Running Simulated Annealing...")
    sa_solution, sa_time, sa_total_time = simulated_annealing(
        initial_solution=initial_solution,
        processing_times=processing_times,
        num_machines=num_machines,
        iterations=iterations,
        show_plots=show_plots
    )

    # Print results
    print(f"Hill Climbing - Best Time: {hc_time:.4f} - Execution Time: {iterations} iterations")
    print(f"Tabu Search - Best Time: {ts_time:.4f} - Execution Time: {iterations} iterations")
    print(f"Simulated Annealing - Best Time: {sa_time:.4f} - Execution Time: {iterations} iterations")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Algorithm Comparison for Open-shop Scheduling')
    parser.add_argument('--jobs', type=int, default=10, help='Number of jobs')
    parser.add_argument('--machines', type=int, default=5, help='Number of machines')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for Hill Climbing and Tabu Search')
    parser.add_argument('--tabu_size', type=int, default=1000, help='Size of tabu list for Tabu Search')
    parser.add_argument('--full_search', type=bool, default=False, help='Test Full Search algorithm for OSS')
    parser.add_argument('--show_plots', type=bool, default=True, help='Show plots during optimization')

    args = parser.parse_args()

    compare_algorithms(args.machines, args.jobs, args.iterations, args.tabu_size, args.show_plots)
    if args.full_search:
        print("Running Full Search...")
        print(f"Brute force best time for 3 machines with 4 tasks: {full_search(3, 4)}")

    time.sleep(30)

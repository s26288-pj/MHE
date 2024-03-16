from random import randint

workload = [1, 3, 5, 2, 8, 1, 2]
devices = [1, 1, 2]


def generate_random_queue(jobs, machines):
    queue = []
    for i in range(len(machines)):
        queue.append([])

    for job in jobs:
        x = randint(0, len(machines) - 1)
        queue[x].append(job)
    return queue


def summarize_job_time(queues):
    sum_job_time = []
    for index, job_time in enumerate(queues):
        full_job_time = 0
        for i in range(len(job_time)):
            full_job_time += job_time[i]
        sum_job_time.append(full_job_time)
    return sum_job_time


def get_maximum_time(time_table):
    return max(time_table)


queues = generate_random_queue(workload, devices)
print(queues)
time_to_finish = summarize_job_time(queues)
print(time_to_finish)
total = get_maximum_time(time_to_finish)
print(total)

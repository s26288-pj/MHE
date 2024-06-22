# Open Shop Scheduling

Open Shop Scheduling, known as OSSP, is an optimization problem in computer science and operations research. It is a variant of task scheduling optimization. In the general task scheduling problem, we have to execute \( n \) tasks \( J1, J2, ..., Jn \) with different processing times on \( m \) machines of varying computational power. The objective is to minimize the makespan, which is the total duration of the schedule (i.e., the time when all tasks are completed processing). In the specific variant known as open-shop scheduling, each task consists of a set of operations \( O1, O2, ..., On \) that must be processed in any order.

## Running the Program from Command Line

To run the program from the command line, use the following commands:

### Hill Climbing and Tabu Search (including Full Search)

```bash
python.exe open_shop_scheduling.py --jobs 100 --machines 20 --iterations 10000 --tabu_size 20

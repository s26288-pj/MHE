# Open Shop Scheduling

Open Shop Scheduling - czyli problem harmonogramowania otwartego (OSSP), to problem optymalizacyjny w informatyce i badaniach operacyjnych. Jest to wariant optymalizacji harmonogramowania zadań. W ogólnym problemie harmonogramowania zadań mamy do wykonania n zadań J1, J2, ..., Jn o różnych czasach przetwarzania, które należy zaplanować na m maszynach o różnej mocy obliczeniowej, starając się jednocześnie zminimalizować czas wykonania - makespan, czyli całkowity czas trwania harmonogramu (czyli czas, gdy wszystkie zadania zostały zakończone przetwarzanie). W specyficznym wariancie znanym jako harmonogramowanie otwarte (open-shop scheduling), każde zadanie składa się z zestawu operacji O1, O2, ..., On, które muszą być przetwarzane w danej kolejności.

## Running the Program from Command Line

To run the program from the command line, use the following commands:

### Hill Climbing and Tabu Search and optionally Brute Force

```bash
python.exe open_shop_scheduling.py --jobs 10 --machines 5 --iterations 1000 --tabu_size 1000 --full_search True
```
### Genethic Algorithm

```bash
python.exe genethic_alghoritm.py --machines 5 --jobs 10 --pop_size 100 --num_generations 100 --num_parents 50 --num_offsprings 50 --mutation_rate 0.1 --crossover_points 2   
```

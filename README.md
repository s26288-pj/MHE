# Open Shop Scheduling

Open Shop Scheduling - czyli problem harmonogramowania otwartego (OSSP), to problem optymalizacyjny w informatyce i badaniach operacyjnych. Jest to wariant optymalizacji harmonogramowania zadań. W ogólnym problemie harmonogramowania zadań mamy do wykonania n zadań J1, J2, ..., Jn o różnych czasach przetwarzania, które należy zaplanować na m maszynach o różnej mocy obliczeniowej, starając się jednocześnie zminimalizować czas wykonania - makespan, czyli całkowity czas trwania harmonogramu (czyli czas, gdy wszystkie zadania zostały zakończone przetwarzanie). W specyficznym wariancie znanym jako harmonogramowanie otwarte (open-shop scheduling), każde zadanie składa się z zestawu operacji O1, O2, ..., On, które muszą być przetwarzane w danej kolejności.

## Running the Program from Command Line

To run the program from the command line, use the following commands:

### Hill Climbing and Tabu Search and optionally Brute Force

```bash
python.exe open_shop_scheduling.py --jobs 100 --machines 20 --iterations 10000 --tabu_size 200 --brute_force True

### Genethic Algorithm

```bash
python.exe genethic_alghoritm.py --machines 10 --jobs 20 --num_generations 100

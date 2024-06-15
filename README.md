Open Shop Scheduling - czyli problem harmonogramowania otwartego (OSSP), to problem optymalizacyjny w informatyce i badaniach operacyjnych. 
Jest to wariant optymalizacji harmonogramowania zadań. W ogólnym problemie harmonogramowania zadań mamy do wykonania n zadań J1, J2, ..., Jn o różnych czasach przetwarzania, które należy zaplanować na m maszynach o różnej mocy obliczeniowej, 
starając się jednocześnie zminimalizować czas wykonania - makespan, czyli całkowity czas trwania harmonogramu (czyli czas, gdy wszystkie zadania zostały zakończone przetwarzanie). 
W specyficznym wariancie znanym jako harmonogramowanie otwarte (open-shop scheduling), każde zadanie składa się z zestawu operacji O1, O2, ..., On, które muszą być przetwarzane w dowolnej kolejności.

**Uruchomienie programu z linii komend**

Hill Climbing oraz Tabu Search (zaimplementowany również full search): python.exe open_shop_scheduling.py --jobs 100 --machines 20 --iterations 10000 --tabu_size 20
Genethic Algorithm: python.exe genethic_alghoritm.py --jobs 500 --machines 20

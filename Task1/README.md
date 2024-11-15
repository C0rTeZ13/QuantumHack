## Установка, запуск и краткая логика работы
1. **Основные файлы для запуска**
   - main.py
   - task-1-stocks.csv
   Эти файлы необходимо загрузить на платформу QBoard (в нашем рабочем пространстве уже добавлены). Далее необходимо запустить main.py.
2. **Краткая логика работы**
   1. Программа предобрабатывает данные: составляет симметричную матрицу Q (QUBO) учитывая ограничения в виде целевого бюджета и целевого риска. Штрафы подобраны эксперементально.
   2. Выполняется qiopt.
   3. После выполнения qiopt на выходе получается массив, представляющий из себя количество купленных акций в первый день для каждой акции в бинарном представлении. Алгоритм преобразует в целочисленное представление.
   4. Далее идёт проверка и фильтрация полученных вариантов портфеля, поиск лучшего варианта соответствующего критериям (критерии подобраны экспериментально).
   5. Если вариант соответствующий критериям не найден, алгоритм сам перезапускается с п.2. Максимальное количество попыток поиска - 10.
3. **Результат работы**
   - Программа выводит номер лучшего найденного варианта портфеля, его доходность, риск.
   - Программа создаёт файл-график (формат .png) для этого портфеля с указанием изменения стоимости по дням (файл - Task1/returns.png).
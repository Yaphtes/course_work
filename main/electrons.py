# Теория:
# https://en.wikipedia.org/wiki/Electron
# https://en.wikipedia.org/wiki/Lorentz_force#Charged_particle
# a = F/m - второй з-н Ньютона
# ! Приближения для изменений скорости и позиции на промежутке dt:
# dv = a*dt
# dx = v*dt

# Как такое движение электронов выглядит в теории:
# https://en.wikipedia.org/wiki/Lorentz_force#/media/File:Charged-particle-drifts.svg

import configparser  # Средства разбора конфиг. файлов
import struct  # Средства упаковки и распаковки двоичных данных
import sys

import numpy as np  # NumPy
from mpi4py import MPI  # Фреймворк MPI

import timer as _timer  # Timer

COMM = MPI.COMM_WORLD
# Идентификатор узла, на котором выполняется данный экземпляр программы
RANK = COMM.Get_rank()
# Количество узлов в системе MPI
SIZE = COMM.Get_size()

# Отображение типов данных с плавающей точкой numpy на типы данных с плавающей точкой MPI
NUMPY_TO_MPI_FP_DATATYPE = {
    np.float32: MPI.FLOAT,
    np.float64: MPI.DOUBLE,
    np.float128: MPI.LONG_DOUBLE
}

# Строковые константы - имена секций в конфиг. файле
CONFIG_GENERAL_PARAMETERS = "General parameters"
CONFIG_VELOCITIES = "Velocities"
CONFIG_POSITIONS = "Positions"
CONFIG_SEPARATOR = "Separator"
CONFIG_E = "E"
CONFIG_B = "B"
CONFIG_T0 = "t0"
CONFIG_T1 = "t1"
CONFIG_DT = "dt"
CONFIG_ELECTRON_COUNT = "#electron_count"
CONFIG_ITERATION_COUNT = "#iter_count"
CONFIG_FP_NUMBERS_FORMAT = "#fp_num_fmt"

# Размерность пространства, в котором проводятся вычисления
DIMENSION = 3


# Главная функция для расчетов, рассчитывает траектории электронов
def calculate(X: np.ndarray, V: np.ndarray, E: np.array, B: np.array,
              t0: float, dt: float, iter_count: int) -> np.ndarray:
    """
    Функция рассчитывает траектории электронов в поле скрещенных сил. Все координаты, скорости и вектора сил полагаются
    трехмерными. Для расчета позиции в момент времени t + dt используется метод Эйлера.

    :param X: Начальные позиции электронов, двумерный numpy массив
    :param V: Начальные скорости электронов
    :param E: Вектор напряженности электрического поля
    :param B: Вектор магнитной индукции
    :param t0: Начальное время расчета
    :param dt: Временной шаг
    :param iter_count: Количество шагов алгоритма.
    :return: Трехмерный numpy массив следующей конфигурации:
        Количество итераций X Количество электронов X Размерность пространства (3),
    содержащий в себе позиции электронов на каждой итерации алгоритма. Тип данных массива будет выведен из
    массива позиций X.
    """
    # Удельный заряд электрона:
    EM = -1.602176565e-19 / 9.10938356e-31
    t = t0
    electron_count = len(X)
    result = np.empty((iter_count, electron_count, DIMENSION), dtype=X.dtype)
    # В указанном промежутке времени
    for j in range(iter_count):
        result[j] = X.copy()
        # Для каждого электрона...
        for i in range(electron_count):
            # Сначала пересчитаем его позицию
            X[i] += V[i] * dt
            # Затем вычислим моментальное ускорение
            a = (E + np.cross(V[i], B)) * EM
            # Затем пересчитаем скорость электрона
            V[i] += a * dt
        # Рассчитав позиции всех электронов, перейдем к следующему моменту времени
        t += dt
    return result


def np_1d_array_from_string(string, separator=" "):
    """
    Преобразует строку в одномерный numpy массив.\n
    :param string: Строка
    :param separator: Разделитель для компонент массива
    :return: numpy массив, содержащий компоненты вектора
    """
    return np.array([float(x) for x in string.split(separator)], dtype=np.float64)


def read_parameters_from_config_file(filename, node=0, numpy_fp_numbers_format=np.float64):
    """
    Вспомогательная функция для разбора конфигурационного файла. Рассчитана на работу с MPI (выполнится только
    на указанном узле).\n
    :param filename: Имя конфиг. файла
    :param node: Узел MPI, на котором хранится файл.
    :param numpy_fp_numbers_format: numpy формат чисел с плавающей точкой
    :return: Параметры, содержащиеся в конфиг. файле в виде ассоциативного массива готовых к использованию данных
    """
    if RANK != node:
        return {}

    config = configparser.ConfigParser()
    config.read(filename)

    result = {}

    general = config[CONFIG_GENERAL_PARAMETERS]

    vector_component_separator = general.get(CONFIG_SEPARATOR, " ")

    result[CONFIG_E] = np_1d_array_from_string(general.get(CONFIG_E))
    result[CONFIG_B] = np_1d_array_from_string(general.get(CONFIG_B))
    result[CONFIG_T0] = t0 = np_1d_array_from_string(general.get(CONFIG_T0))
    result[CONFIG_T1] = t1 = np_1d_array_from_string(general.get(CONFIG_T1))
    result[CONFIG_DT] = dt = np_1d_array_from_string(general.get(CONFIG_DT))
    result[CONFIG_ITERATION_COUNT] = int((t1 - t0) / dt)
    result[CONFIG_FP_NUMBERS_FORMAT] = numpy_fp_numbers_format

    for cfg_property in (CONFIG_POSITIONS, CONFIG_VELOCITIES):
        conf = config[cfg_property]
        results = []
        if "raw" in conf:
            lines = conf["raw"].split("\n")
            for line in lines:
                if line:
                    results.append([float(x) for x in line.split(vector_component_separator)])
        elif "file" in conf:
            content_file = conf["file"]
            bin_mode = "b" if "binary" in conf and conf.getboolean("binary") else ""
            with open(content_file, "r" + bin_mode) as file:
                if bin_mode:
                    fmt = "%dd" % DIMENSION
                    size = struct.calcsize(fmt)
                    while True:
                        result = file.read(size)
                        if len(result) < size:
                            break
                        # noinspection PyTypeChecker
                        results.append(struct.unpack(fmt, result))
                else:
                    for line in file.readlines():
                        if line:
                            results.append([float(x) for x in line.split(vector_component_separator)])
        else:
            if cfg_property == CONFIG_POSITIONS:
                raise ValueError("Can't find initial values for electrons: " + cfg_property)
            result[cfg_property] = np.zeros((len(result[CONFIG_POSITIONS]), DIMENSION), dtype=numpy_fp_numbers_format)
        result[cfg_property] = np.array(results, dtype=numpy_fp_numbers_format)
    result[CONFIG_ELECTRON_COUNT] = len(result[CONFIG_POSITIONS])
    return result


def compute_part_lengths(elements_count, shape=(None, 3)):
    """
    Вспомогательная функция для равномерного распределения количества данных на все узлы.\n
    :param elements_count: количество элементов
    :param shape: Форма распределяемой матрицы, первый аргумент игнорируется
    :return: Список длиной SIZE, содержащий количества электронов для каждого узла;
        список длиной SIZE, содержащий длины частей массива начальных данных, которые будут
        разосланы на узлы, в элементарных единицах (т.е. в количестве чисел);
        список длиной SIZE, содержащий смещения частей массива начальных данных в элементарных единицах
    """
    rough, rem = elements_count // SIZE, elements_count % SIZE
    mul = 1
    for i in range(len(shape) - 1):
        mul *= shape[i + 1]
    if rem == 0:
        return [rough] * SIZE, \
               [rough * mul] * SIZE, \
               [rough * i * mul for i in range(SIZE)]
    return [(rough + 1)] * rem + [rough] * (SIZE - rem), \
           [(rough + 1) * mul] * rem + [rough * mul] * (SIZE - rem), \
           [(rough + 1) * mul * i for i in range(rem)] + [rem * mul + (rough * i * mul) for i in range(rem, SIZE)]


# Общая идея параллелизации:
# 1. На корневом узле (далее хост) считаем все необходимые параметры, такие как начальные позиции и скорости электронов,
# из конфигурационного файла;
# 2. Равномерно распределим эти данные между узлами, выделив каждому узлу некоторое количество электронов для расчета.
# Если количество электронов не кратно количеству узлов, то распределим остаток K между первыми K узлами;
# 3. Получив данные, произведем на каждом узле расчет траекторий;
# 4. На хосте соберем результат в единый массив, выполнив при этом, если потребуется, его реорганизацию.
def parallel_run(parameters: dict) -> tuple:
    """
    Главная функция для параллелизации расчета.\n
    :param parameters: Ассоциативный массив параметров, необходимых для расчета
    :return: Кортеж, состоящий из результата, представленного в виде многомерного numpy массива
    и кортежа таймеров (timer.Timer) следующего вида:
    (
        Таймер инициализации,
        [Таймер расчета на 0 узле,
         Таймер расчета на 1 узле,
         ...
         Таймер расчета на SIZE-м узле],
        Таймер сбора и постобработки информации
    )
    """
    # Таймер процесса инициализации
    host_scatter_timer = None if RANK != 0 else _timer.Timer().start()
    # Распространим параметры, необходимые для инициализации, на все узлы
    # Используемый формат чисел с плавающей точкой для numpy
    numpy_fp_numbers_format = COMM.bcast(None if RANK != 0 else parameters[CONFIG_FP_NUMBERS_FORMAT])
    # Используемый формат чисел с плавающей точкой для mpi
    mpi_fp_numbers_format = NUMPY_TO_MPI_FP_DATATYPE[numpy_fp_numbers_format]
    # Полное количество электронов
    host_electron_count = None if RANK != 0 else parameters[CONFIG_ELECTRON_COUNT]
    # Данные о распределении электронов
    node_electron_count_list, host_part_lengths, host_displacements = \
        (None, None, None) if RANK != 0 else compute_part_lengths(host_electron_count)

    # Распространим информацию о распределении электронов на все узлы
    node_electron_count = COMM.scatter(node_electron_count_list)

    # Инициализируем пустые массивы для позиций и скоростей на каждом узле
    X = np.empty((node_electron_count, DIMENSION), dtype=numpy_fp_numbers_format)
    V = np.empty((node_electron_count, DIMENSION), dtype=numpy_fp_numbers_format)

    # Распространим начальные данные на все узлы:
    # Начальные координаты
    COMM.Scatterv(None if RANK != 0 else [parameters.get(CONFIG_POSITIONS, None),
                                          host_part_lengths,
                                          host_displacements,
                                          mpi_fp_numbers_format], X)
    # Начальные скорости
    COMM.Scatterv(None if RANK != 0 else [parameters.get(CONFIG_VELOCITIES, None),
                                          host_part_lengths,
                                          host_displacements,
                                          mpi_fp_numbers_format], V)
    # Вектора полей и временнЫе параметры
    E, B, t0, dt, iter_count = COMM.bcast(None if RANK != 0 else
                                          [parameters[CONFIG_E],
                                           parameters[CONFIG_B],
                                           parameters[CONFIG_T0],
                                           parameters[CONFIG_DT],
                                           parameters[CONFIG_ITERATION_COUNT]])
    # По умолчанию, iter_count - это количество итераций, необходимое для того чтобы рассчитать
    # траектории в интервале времени [t0, t1). Добавим в этот интервал точку t1:
    iter_count += 1

    # Инициализация окончена, остановим таймер
    if RANK == 0:
        host_scatter_timer.stop()

    # В этой точке у нас есть все данные, соответствующие узлу, на котором мы выполняемся
    # Замерим время расчета: стартуем таймер
    local_calculation_timer = _timer.Timer().start()
    # Сам расчет:
    part_of_result = calculate(X, V, E, B, t0, dt, iter_count)
    # Остановим таймер расчета
    local_calculation_timer.stop()

    # В этой точке расчет траекторий окончен, вернем данные хосту
    host_gather_timer = None
    result = None
    host_output_part_lengths = None
    host_output_part_displacements = None
    if RANK == 0:
        host_gather_timer = _timer.Timer().start()
        # Одномерный массив, являющийся буфером для данных со всех узлов
        result = np.empty((iter_count * host_electron_count * DIMENSION,), dtype=numpy_fp_numbers_format)
        # Длины результатов с каждого узла
        host_output_part_lengths = [iter_count * x for x in host_part_lengths]
        # Смещения результатов с каждого узла
        host_output_part_displacements = [iter_count * x for x in host_displacements]
    # Собираем результат
    COMM.Gatherv(part_of_result, None if RANK != 0 else [result,
                                                         host_output_part_lengths,
                                                         host_output_part_displacements,
                                                         mpi_fp_numbers_format])
    # Собираем таймеры с узлов
    local_calculation_timer_list = COMM.gather(local_calculation_timer)
    # На хосте производим финальную обработку данных
    if RANK == 0:
        # Как было показано в процессе инициализации, количество электронов распределено по узлам неравномерно.
        # Всего существует 2 типа узлов: те, на которых оказалось k = (Количество электронов)/(Количество узлов) + 1
        # электронов, и те, на которых были расчитаны траектории для k - 1 электронов, причем сначала идут данные
        # с узлов 1-го типа, а потом - с узлов 2-го.
        # Количество электронов на узле 1-го и 2-го типов:
        part_len_former, part_len_latter = node_electron_count_list[0], node_electron_count_list[-1]
        size_former = node_electron_count_list.count(part_len_former)
        size_latter = len(node_electron_count_list) - size_former
        # Размер части массива, занятой данными узлов 1-го типа (в элементарных единицах):
        distr_part_size = DIMENSION * (part_len_former * size_former) * iter_count
        # Если электроны равномерно распределены по узлам, либо (Количество электронов) < SIZE
        if part_len_former == part_len_latter or part_len_latter == 0:
            size = SIZE if part_len_latter != 0 else host_electron_count
            # Для получения массива формы (Количество итераций)X(Количество электронов)X(DIMENSION)
            # проделаем следующие преобразования:
            # 1. Преобразуем одномерный массив данных к четырехмерному массиву формы
            # (Количество узлов)X(Количество итераций)X(Количеcтво электронов на узле)X(DIMENSION)
            # 2. Поменяем местами 1 и 2 измерения
            # 3. Объединим 2 и 3 измерения
            result = result \
                .reshape((size, iter_count, part_len_former, DIMENSION,)) \
                .transpose((1, 0, 2, 3)) \
                .reshape((iter_count, part_len_former * size, DIMENSION))
        else:
            # Результат состоит из двух условных частей.
            # Пример для 3х электронов, 2х узлов, 3х итераций:
            # [1, 2, 1, 2, 1, 2, 3, 3, 3]
            # Последовательно выполним следующие преобразования:
            # 1) Явно разделим массив на 2
            # [1, 2, 1, 2, 1, 2], [3, 3, 3]
            # 2) Преобразуем массивы к двумерным
            # [[1, 2],      [[3],
            #  [1, 2],       [3],
            #  [1, 2]]       [3]]
            # 3) Объединим их горизонтально (np.hstack)
            # [[1, 2, 3],
            #  [1, 2, 3],
            #  [1, 2, 3]]
            result = np.hstack((
                result[:distr_part_size].reshape((iter_count, size_former * part_len_former, DIMENSION)),
                result[distr_part_size:].reshape((iter_count, size_latter * part_len_latter, DIMENSION))
            ))
        # В этой точке мы имеем массив данных об электронах, точно такой же,
        # как если бы мы считали все на одной машине.
        return result, (host_scatter_timer, local_calculation_timer_list, host_gather_timer.stop())
    return None, None


# Обработка аргументов из командной строки
def parse_command_line_args(argv):
    opt = []
    par = []
    for arg in argv:
        if arg.startswith("-"):
            opt.append(arg)
        else:
            par.append(arg)
    return opt, par

# Тело программы (main)
if __name__ == "__main__":
    # Замерим полное время выполнения программы: стартуем таймер
    full_time_timer = _timer.Timer().start()
    if len(sys.argv) < 2:
        print("too few args, exiting...")
        exit()

    opt, par = parse_command_line_args(sys.argv[1:])

    # В каком формате вывести отчет о выполнении - сжатом (указан аргумент -testperf) или полном (арг. не указан)
    use_test_format = "-testperf" in opt

    # Считаем параметры из конфиг. файла
    parameters = read_parameters_from_config_file(par[0])
    # Произведем расчет
    result, timings = parallel_run(parameters)
    if result is None:
        # Если мы в этой точке, то это значит, что мы выполняемся не на главном узле и можно завершить выполнение
        exit()

    # Если в параметрах указано имя файла - сохраним данные в файл
    save_timer = None
    if len(par) >= 2:
        # Замерим время сохранения в файл
        save_timer = _timer.Timer().start()
        np.save(par[1], result)
        save_timer.stop()
    # Остановим таймер полного времени
    full_time_timer.stop()

    # Выведем отчет
    if use_test_format:
        # В тестовом формате нам важно только кол-во используемых узлов, кол-во электронов, кол-во итераций и
        # затраченное на расчет время
        print(SIZE, parameters[CONFIG_ELECTRON_COUNT], parameters[CONFIG_ITERATION_COUNT], full_time_timer.t)
        exit()

    # Если мы не в тестовом режиме, имеет смысл вывести полный отчет по проведенному расчету
    print("-- Execution finished successfully --- ")
    print("Electron count:", parameters[CONFIG_ELECTRON_COUNT])
    print("Iteration count:", parameters[CONFIG_ITERATION_COUNT])
    print("\t time range: [%g:%g], dt=%g" % (parameters[CONFIG_T0], parameters[CONFIG_T1], parameters[CONFIG_DT]))
    print("Summary time:", full_time_timer.get_str())
    print("Time spent on:")
    print("\tInitialization (root node):", timings[0].get_str())
    print("\tComputation:")
    for i in range(SIZE):
        print("\t\tNode %d: %s" % (i, timings[1][i].get_str()))
    print("\tGathering results (root node):", timings[2].get_str())
    print("\tSaving results to file:", save_timer.get_str() if save_timer else None)

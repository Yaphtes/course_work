# Теория:
# https://en.wikipedia.org/wiki/Electron
# https://en.wikipedia.org/wiki/Lorentz_force#Charged_particle
# a = F/m - второй з-н Ньютона
# ! Приближения для изменений скорости и позиции на промежутке dt:
# dv = a*dt
# dx = v*dt

# Как такое движение электронов выглядит в теории:
# https://en.wikipedia.org/wiki/Lorentz_force#/media/File:Charged-particle-drifts.svg

import configparser
import os
import sys

import numpy as np
from mpi4py import MPI

sys.path.append(os.curdir)
import timer as _timer

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

CONFIG_GENERAL_PARAMETERS = "General parameters"
CONFIG_VELOCITIES = "Velocities"
CONFIG_POSITIONS = "Positions"

M = 9.10938356e-31
E = -1.602176565e-19

# Удельный заряд электрона:
EM = E / M


def calculate(X, V, E, B, t0, dt, iter_count):
    global EM
    t = t0
    N = len(X)
    # Estimate size of result:
    result = np.empty((iter_count, N, X.shape[1]), dtype=np.float64)
    # В указанном промежутке времени
    for j in range(iter_count):
        result[j] = X.copy()
        # Для каждого электрона
        for i in range(N):
            # Сначала пересчитаем его позицию, используя скорость
            X[i] += V[i] * dt
            # Затем вычислим моментальное ускорение
            a = (E + np.cross(V[i], B)) * EM
            # Затем пересчитаем скорость электрона, используя ускорение
            V[i] += a * dt
        t += dt
    return result


def test_calculate():
    # Задаем начальные позиции двух электронов
    X = np.array([(1, 0, 0),
                  (0, 1, 0),
                  ],
                 dtype=np.float64)
    # Задаем их начальные скорости
    V = np.array([(0, 1e7, 0),  # 10**7 м/с вдоль оси y
                  (1e7, 0, 0),  # 10**7 м/с вдоль оси x
                  ],
                 dtype=np.float64)
    # Задаем векторы E и B
    E = np.array([0, 0, 0])
    B = np.array([0, 0, 1e-3])
    # ! ВАЖНО ! Электроны движутся с огромными скоростями, поэтому для кореектного расчета
    # величина dt должна быть малой. Здесь во временном промежутке длиной в 1 наносекунду
    # расчет проводится с шагом в 0.01 наносекунды, т.е. 100 значений
    t0, tf, dt = 0, 1e-9, 1e-11
    # Считаем
    result = calculate(X, V, E, B, t0, tf, dt)
    print(*result, sep="\n")


def np_1d_array_from_string(string):
    return np.array([float(x) for x in string.split(" ")], dtype=np.float64)


def read_parameters_from_config_file(filename):
    curdir = os.path.split(filename)[0]  # os.curdir +

    config = configparser.ConfigParser()
    config.read(filename)

    dimensions = 3

    result = dict()

    general = config[CONFIG_GENERAL_PARAMETERS]
    result["E"] = np_1d_array_from_string(general.get("E"))
    result["B"] = np_1d_array_from_string(general.get("B"))
    result["t0"] = np_1d_array_from_string(general.get("t0"))
    result["t1"] = np_1d_array_from_string(general.get("t1"))
    result["dt"] = np_1d_array_from_string(general.get("dt"))

    for cfg_property in (CONFIG_POSITIONS, CONFIG_VELOCITIES):
        conf = config[cfg_property]
        results = []
        if "raw" in conf:
            lines = conf["raw"].split("\n")
            for line in lines:
                if line:
                    results.append([float(x) for x in line.split(" ")])
        elif "file" in conf:
            fname = conf["file"]
            binr = "b" if "binary" in conf and conf.getboolean("binary") else ""
            with open(  # curdir + os.path.sep +
                    fname, "r" + binr) as file:
                if binr:
                    import struct
                    fmt = "%dd" % dimensions
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
                            results.append([float(x) for x in line.split(" ")])
        else:
            if cfg_property == CONFIG_POSITIONS:
                raise ValueError("Can't find initial values for electrons: " + cfg_property)
            result[cfg_property] = np.zeros((len(result[CONFIG_POSITIONS]), dimensions), dtype=np.float64)
        result[cfg_property] = np.array(results, dtype=np.float64)

    return result


# Дан вычислительный кластер из M узлов
# 1) На главном узле задаются начальные данные для N электронов в виде
# некоторого массива DATA
# 2) Каждый узел обсчитывает K (K = N / M) электронов
# 3) Главная узел собирает данные
def parallel_run(config_file_name):
    parameters = dict() if RANK != 0 else read_parameters_from_config_file(config_file_name)

    X_host = None
    V_host = None

    if RANK == 0:
        # Начальные позиции
        X_host = parameters[CONFIG_POSITIONS]
        # Начальные скорости
        V_host = parameters[CONFIG_VELOCITIES]

    part_len, dim_ = COMM.bcast(None if RANK != 0 else (len(X_host) // SIZE, X_host.shape[1]))

    X = np.empty((part_len, dim_), dtype=np.float64)
    V = np.empty((part_len, dim_), dtype=np.float64)

    # Распространим начальные данные на все узлы
    COMM.Scatter(X_host, X)
    COMM.Scatter(V_host, V)
    E, B, t0, tf, dt = COMM.bcast(None if RANK != 0 else
                                  [parameters.get("E"), parameters.get("B"),
                                   parameters.get("t0"), parameters.get("t1"), parameters.get("dt")])

    # +1 - значит мы хотим получить интервал
    points_count = int((tf - t0) / dt) + 1

    # В этой точке у нас есть все данные, и мы можем начать расчет
    part_of_result = calculate(X, V, E, B, t0, dt, points_count)

    # Теперь вернем данные хосту
    result = None
    if RANK == 0:
        result = np.empty((SIZE, points_count, part_len, dim_), dtype=np.float64)

    COMM.Gather(part_of_result, result, root=0)

    if RANK == 0:
        # Теперь у нас есть массив данных об электронах, но он разбит на массивы
        # согласно узлам. Объединим эти массивы
        result = result.transpose((1, 0, 2, 3))
        result = result.reshape((points_count, part_len * SIZE, dim_))
        # В этой точке мы имеем массив данных об электронах, точно такой же,
        # как если бы мы считали все на одной машине
        return result
    else:
        print("Non-host process %d completed..." % RANK)
        return None


def compute_perfomance():
    timer = _timer.Timer()

    timer.start()
    parallel_run("config.cfg")
    timer.stop()
    print("Completed in %d ms" % timer.get_ms())

    # TODO: add more perfomance computing points


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("too few args, exiting...")
        exit()

    result = parallel_run(sys.argv[1])
    if result is None:
        exit()

    if len(sys.argv) > 2:
        output_file_name = sys.argv[2]
        np.save(output_file_name, result)
        print("Result saved to file: " + output_file_name)

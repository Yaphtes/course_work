import configparser
import random
import struct
import sys

CONFIG_FILE_TEMPLATE = u"""
# В этой секции указываются основные параметры расчета
[General parameters]
# Вектор электрического поля
E = {E}
# Вектор магнитного поля
B = {B}
# Временной промежуток: от t0 до t1 с шагом dt
t0 = {t0}
t1 = {t1}
dt = {dt}


# В этой секции указываются начальные положения электронов
[Positions]
# Вектора положений в чистом виде. Отступ перед значением обязателен.
#raw =
#	0 1 0
#	0 1.1 0
#	0 1.2 0
#	0 1.3 0
# Путь к файлу с векторами положений (относительно текущей директории)
file = {pf}
# Файл двоичный?
#binary = {isbin_pos}


# В этой секции указываются начальные скорости электронов. Если секция пуста, то скорости
# принимаются равными (0, 0, 0).
[Velocities]
# Вектора скоростей в чистом виде. Отступ перед значением обязателен.
#raw =
#   1e7 0 0
#   1e7 1e7 0
#   1e6 0 0
#   0 1e7 0
# Путь к файлу с векторами скоростей (относительно текущей директории)
file = {vf}
# Файл двоичный?
#binary = {isbin_vel}
"""


def create_cfg(name, parameters):
    with open(name, "w", encoding='utf-8') as file:
        file.writelines(CONFIG_FILE_TEMPLATE.format(**parameters))


def parse_cfg(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    result = {}

    g = config["General"]
    result['n'] = g.getint('electrons count')
    result['cfg'] = g['output config file']
    result['isbin_vel'] = g.getboolean('binary velocities output file')
    result['vf'] = g['velocities output file']
    result['isbin_pos'] = g.getboolean('binary positions output file')
    result['pf'] = g['positions output file']

    result['E'] = g['E']
    result['B'] = g['B']
    result['t0'] = g['t0']
    result['t1'] = g['t1']
    result['dt'] = g['dt']

    pb = config["Positions bounds"]
    pb_x = (pb.getfloat('x min'), pb.getfloat('x max'))
    pb_y = (pb.getfloat('y min'), pb.getfloat('y max'))
    pb_z = (pb.getfloat('z min'), pb.getfloat('z max'))
    result['pb'] = pb_x, pb_y, pb_z

    vb = config["Velocities bounds"]
    vb_x = (vb.getfloat('x min'), vb.getfloat('x max'))
    vb_y = (vb.getfloat('y min'), vb.getfloat('y max'))
    vb_z = (vb.getfloat('z min'), vb.getfloat('z max'))
    result['vb'] = vb_x, vb_y, vb_z

    config.clear()

    return result


def rnd(a, b):
    return random.random() * (b - a) + a


def create_data(n, xrange: tuple, yrange: tuple, zrange: tuple = None):
    return [(rnd(xrange[0], xrange[1]),
             rnd(yrange[0], yrange[1]),
             0 if not zrange else rnd(zrange[0], zrange[1])) for i in range(n)]


def write_file(filename, data, binary=False):
    if binary:
        with open(filename, "wb") as file:
            for data_value in data:
                file.write(struct.pack("3d", data_value))
    else:
        with open(filename, "w") as file:
            file.writelines([" ".join([str(d0) for d0 in d]) + "\n" for d in data])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("too few arguments, exiting...")
        exit()

    # print(sys.argv)
    config = parse_cfg(sys.argv[1])

    write_file(config['pf'], create_data(config['n'], *config['pb']), binary=config["isbin_pos"])
    write_file(config['vf'], create_data(config['n'], *config['vb']), binary=config["isbin_vel"])
    create_cfg(config['cfg'], config)

    print(config['cfg'], sep="", end="")

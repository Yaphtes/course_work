import random
import struct
import sys

def rnd(a, b):
    return random.random() * (b - a) + a


def create_data(n, xrange:tuple, yrange:tuple, zrange:tuple=None):
    return [(rnd(xrange[0], xrange[1]), rnd(yrange[0], yrange[1]), 0 if not zrange else rnd(zrange[0], zrange[1])) for i in range(n)]


def write_file(filename, data, binary=False):
    if binary:
        with open(filename, "wb") as file:
            for data_value in data:
                file.write(struct.pack("3d", data_value))
    else:
        with open(filename, "w") as file:
            file.writelines([" ".join([str(d0) for d0 in d]) + "\n" for d in data])

if __name__ == '__main__':

    n = 100
    write_file("..\\positions1.txt", create_data(n, (0, 1), (0, 1), (0, 0)))
    write_file("..\\velocities1.txt", create_data(n, (1e6, 1e7), (-1e7, -1e6), (0, 0)))
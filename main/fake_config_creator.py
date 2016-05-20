import configparser
import random
import struct
import sys


def parse_cfg(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    g = config["General"]
    n = g.getint('electrons count')
    isbinv = g.getboolean('binary velocities output file')
    velocities_file = g['velocities output file']
    isbinp = g.getboolean('binary positions output file')
    positions_file = g['positions output file']

    pb = config["Positions bounds"]
    pb_x = (pb['x min'], pb['x max'])
    pb_y = (pb['y min'], pb['y max'])
    pb_z = (pb['z min'], pb['z max'])
    pb_bounds = pb_x, pb_y, pb_z

    vb = config["Velocities bounds"]
    vb_x = (vb['x min'], vb['x max'])
    vb_y = (vb['y min'], vb['y max'])
    vb_z = (vb['z min'], vb['z max'])
    vb_bounds = vb_x, vb_y, vb_z

    config.clear()

    return {"n": n, "pf": positions_file, "pb": pb_bounds, "vf": velocities_file, "vb": vb_bounds,
            "isbin_vel": isbinv, "isbin_pos": isbinp}


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

    config = parse_cfg(sys.argv[1])

    write_file(config['pf'], create_data(config['n'], *config['pb']), binary=config["isbinp"])
    write_file(config['vf'], create_data(config['n'], *config['vb']), binary=config["isbinv"])

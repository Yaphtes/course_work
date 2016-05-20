import matplotlib.pyplot as pp
import numpy as np
import sys
import timer

import matplotlib.animation as animation


def str_color(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def load_from_file(filename):
    result = None
    with open(filename, "rb") as file:
        result = np.load(file)
    return result


class Electron:
    def __init__(self, plt, color):
        self.line = plt.plot([], [], "r-")[0]
        self.line.set_color(color)
        self.point = plt.plot([], [], "r.")[0]
        self.point.set_color(color)


i = 0
def update_lines(num, electrons):
    global i, result
    j = 0
    for electron in electrons:
        # VERY SLOW
        electron.line.set_xdata(np.append(electron.line.get_xdata(), result[i][j][0]))
        electron.line.set_ydata(np.append(electron.line.get_ydata(), result[i][j][1]))
        electron.point.set_xdata(result[i][j][0])
        electron.point.set_ydata(result[i][j][1])
        j += 1
    i += 1


def display_animation(result, coords):
    figure = pp.figure()

    pp.xlim(-coords[0] + coords[2], coords[0] + coords[2])
    pp.ylim(-coords[1] + coords[3], coords[1] + coords[3])


    N = len(result[0])
    frames = len(result)
    electrons = [Electron(pp, str_color(255 * (i / N), 0, 255 * (1 - (i / N)))) for i in range(N)]
    line_ani = animation.FuncAnimation(figure, update_lines, fargs=(electrons,), frames=1000)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=2400)
    t = timer.Timer()
    t.start()
    print("starting writing %d frames" % frames)
    line_ani.save("D:\\line.mp4", writer=writer)
    t.stop()
    print(t.get_ms())
    try:
        #pp.show()
        pass
    except AttributeError:
        pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("no file specified, exiting")
        exit()
    result = load_from_file(sys.argv[1])

    display_animation(result, (0.5, 0.5, 0.5, 0.5))
import subprocess
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as pp
import numpy as np


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


def update(i, result, electrons):
    j = 0
    for electron in electrons:
        electron.line.set_xdata(np.append(electron.line.get_xdata(), result[i][j][0]))
        electron.line.set_ydata(np.append(electron.line.get_ydata(), result[i][j][1]))
        electron.point.set_xdata(result[i][j][0])
        electron.point.set_ydata(result[i][j][1])
        j += 1


def save_animation_in_subprocess(filename, figure, frames_count, fps, bitrate, update, args):
    canvas_width, canvas_height = figure.canvas.get_width_height()
    # Open an ffmpeg process
    cmdstring = ('ffmpeg',
                 '-y', '-r', str(fps),  # overwrite, fps
                 '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                 '-pix_fmt', 'argb',  # format
                 '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                 '-b:v', '%dk' % bitrate,
                 '-minrate', '%dk' % bitrate,
                 '-maxrate', '%dk' % bitrate,
                 '-bufsize', '%dk' % (bitrate * 3),
                 '-vcodec', 'mpeg4', filename)  # output encoding
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    # Draw frames and write to the pipe
    i = 0
    for frame in range(frames_count):
        # draw the frame
        update(i, *args)
        figure.canvas.draw()
        p.stdin.write(figure.canvas.tostring_argb())
        i += 1
    p.communicate()


def display_animation(result, coords, filename=None, save=True,
                      use_subprocess=False, show_plot=False, fps=30, bit_rate=2400):
    figure = pp.figure()

    pp.xlim(-coords[0] + coords[2], coords[0] + coords[2])
    pp.ylim(-coords[1] + coords[3], coords[1] + coords[3])

    electron_count = len(result[0])
    frames = len(result)
    # noinspection PyTypeChecker
    electrons = [Electron(pp, str_color(255 * (i / electron_count),
                                        0,
                                        255 * (1 - (i / electron_count)))) for i in range(electron_count)]
    if use_subprocess and save:
        save_animation_in_subprocess(filename, figure, frames, fps, bit_rate, update, args=(result, electrons,))
        if show_plot:
            raise Exception("Please consider using use_subprocess=False in order to show plot")
    else:
        line_ani = animation.FuncAnimation(figure, update, fargs=(result, electrons,), frames=frames, repeat=False)
        if save:
            writer = animation.writers['ffmpeg'](fps=fps, bitrate=bit_rate)
            line_ani.save(filename, writer=writer)
        if show_plot:
            try:
                pp.show()
            except AttributeError:
                pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("no file specified, exiting")
        exit()
    result = load_from_file(sys.argv[1])

    display_animation(result, (0.5, 0.5, 0.5, 0.5), filename="result.mp4", use_subprocess=True)

import subprocess
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as pp
import numpy as np


def str_color(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def load_from_file(filename):
    with open(filename, "rb") as file:
        return np.load(file)


def _save_animation_subprocess(filename, figure, frames_count, fps, bitrate, update_fn, args):
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
        update_fn(i, *args)
        figure.canvas.draw()
        p.stdin.write(figure.canvas.tostring_argb())
        i += 1
    p.communicate()


def _save_animation_subprocess_parallel(filename, figure, frames_count, fps, bitrate, update_fn, args):
    pass


def _save_animation_matplotlib(filename, figure, frames_count, fps, bitrate, update_fn, args):
    line_ani = animation.FuncAnimation(figure, update_fn, fargs=args, frames=frames_count, repeat=False)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
    line_ani.save(filename, writer=writer)


SAVE_METHOD = {"default": _save_animation_matplotlib,
               "subprocess": _save_animation_subprocess,
               "parallel": _save_animation_subprocess_parallel}


def update_fn(i, result, electrons):
    j = 0
    for electron in electrons:
        electron.line.set_xdata(np.append(electron.line.get_xdata(), result[i][j][0]))
        electron.line.set_ydata(np.append(electron.line.get_ydata(), result[i][j][1]))
        electron.point.set_xdata(result[i][j][0])
        electron.point.set_ydata(result[i][j][1])
        j += 1


def save_animation(result, xrange, yrange, filename, electron_color_fn=None, method="subprocess", fps=30,
                   bit_rate=2400):
    figure = pp.figure()
    pp.xlim(xrange[0], xrange[1])
    pp.ylim(yrange[0], yrange[1])
    electron_count, frame_count = len(result[0]), len(result)
    if electron_color_fn is None:
        electron_color_fn = lambda i, c: "r"
    # noinspection PyTypeChecker
    electrons = [Electron(pp, electron_color_fn(i, electron_count)) for i in range(electron_count)]
    method_fn = SAVE_METHOD.get(method, None)
    if not method_fn:
        raise Exception("No such method: %s" % method)
    # noinspection PyCallingNonCallable
    method_fn(filename, figure, frame_count, fps, bit_rate, update_fn, args=(result, electrons,))


def display_animation(result, xrange, yrange, electron_color_fn=None):
    figure = pp.figure()
    pp.xlim(xrange[0], xrange[1])
    pp.ylim(yrange[0], yrange[1])
    electron_count = len(result[0])
    frame_count = len(result)
    if electron_color_fn is None:
        electron_color_fn = lambda i, c: "r"
    # noinspection PyTypeChecker
    electrons = [Electron(pp, electron_color_fn(i, electron_count)) for i in range(electron_count)]
    line_ani = animation.FuncAnimation(figure, update_fn, fargs=(result, electrons,), frames=frame_count, repeat=False)
    try:
        pp.show()
    except AttributeError:
        pass


class Electron:
    def __init__(self, plt, color):
        self.line = plt.plot([], [], "r-")[0]
        self.line.set_color(color)
        self.point = plt.plot([], [], "r.")[0]
        self.point.set_color(color)


def sample_color_function(i, size):
    return str_color(int(255 * (i / size)), 0, int(255 * (1 - i / size)))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("no output and input file specified, exiting")
        exit()
    result = load_from_file(sys.argv[1])
    filename = sys.argv[2]

    argc = len(sys.argv)
    method = "subprocess" if argc < 4 else sys.argv[3]
    xrange = (0, 1) if argc < 5 else sys.argv[4].split(",")
    yrange = (0, 1) if argc < 6 else sys.argv[5].split(",")

    save_animation(result, xrange, yrange, filename,
                   electron_color_fn=sample_color_function,
                   method=method)

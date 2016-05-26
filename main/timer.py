import time


class Timer:

    def __init__(self):
        self.t = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        self.stopped = False
        self.t = time.perf_counter()
        return self

    def stop(self):
        self.started = False
        self.stopped = True
        self.t = time.perf_counter() - self.t
        return self

    def get_s(self):
        return round(self.t)

    def get_ms(self):
        return round(self.t * 1e3)

    def get_us(self):
        return round(self.t * 1e6)

    def get_ns(self):
        return round(self.t * 1e9)

    def get_str(self):
        return "%d m, %d s, %d ms, %d us" % (
            self.get_s() // 60,
            self.get_s() % 60,
            self.get_ms() % 1e3,
            self.get_us() % 1e3,
        )

    def __repr__(self):
        return "Timer<started: {}, stopped: {}, t: {}>".format(
            self.started, self.stopped, self.get_str()
        )

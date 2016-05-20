import time


class Timer:

    def __init__(self):
        self.t = 0

    def start(self):
        self.t = time.perf_counter()

    def stop(self):
        self.t = time.perf_counter() - self.t

    def get_s(self):
        return round(self.t)

    def get_ms(self):
        return round(self.t * 1e3)

    def get_us(self):
        return round(self.t * 1e6)

    def get_ns(self):
        return round(self.t * 1e9)
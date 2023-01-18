import numpy as np


def for_one_polarization(fn):
    def wrapped(self, x):
        assert x.ndim == 1
        return fn(self, x)

    return wrapped


def for_two_polarizations(fn):
    def wrapped(self, x):
        assert x.ndim == 2
        return fn(self, x)

    return wrapped


class Dispatcher:
    def __init__(self) -> None:
        self.one = None
        self.two = None

    def for_one(self, fn) -> None:
        assert self.one is None
        self.one = fn

    def for_two(self, fn) -> None:
        assert self.two is None
        self.two = fn

    def __call__(self, s, x):
        print(self, s, x)
        if x.ndim == 1:
            assert self.one is not None
            return self.one(s, x)
        if x.ndim == 2:
            assert self.two is not None
            return self.two(s, x)
        assert False


def for_polarization(fn):
    # def dispatcher(self, x):
    #     return dispatcher._fns[x.ndim](self, x)

    # def for_one(fn):
    #     dispatcher._fns[1] = fn

    # def for_two(fn):
    #     dispatcher._fns[2] = fn

    # dispatcher._fns = {}
    # dispatcher.for_one = for_one
    # dispatcher.for_two = for_two

    # return dispatcher
    return Dispatcher()


class A:
    @for_polarization
    def __call__(self, x):
        assert False

    @__call__.for_one
    def impl1(self, x):
        print(f"Hello {x.shape}")

    @__call__.for_two
    def impl2(self, x):
        print(f"Hello {x.shape}")


if __name__ == "__main__":
    a = A()
    print(a.__call__.__dict__)
    A.__call__(a, np.arange(3))
    a(np.zeros((4, 2)))
    a(np.zeros(6))

from abc import abstractmethod

import numpy as np

from utils import Component


class Channel(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    @abstractmethod
    def __call__(self, symbols: np.ndarray) -> np.ndarray:
        assert symbols.ndim == 1


class AWGN(Channel):
    def __init__(self, N0: float) -> None:
        super().__init__()

        self.N0 = N0
        self.rng = np.random.default_rng()

    def __call__(self, symbols: np.ndarray) -> np.ndarray:
        super().__call__(symbols)

        # normal() takes the standard deviation.
        noise = self.rng.normal(0, np.sqrt(self.N0 / 2), size=symbols.size)
        return symbols + noise

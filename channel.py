from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from utils import Component, signal_power


class Channel(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    @abstractmethod
    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert symbols.ndim == 1


class AWGN(Channel):
    def __init__(self, es_n0: float, samples_per_symbol: int) -> None:
        super().__init__()

        assert es_n0 > 0
        self.es_n0 = es_n0

        assert samples_per_symbol > 0
        self.samples_per_symbol = samples_per_symbol

        self.rng = np.random.default_rng()

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        super().__call__(symbols)

        # Each symbol can consist of multiple samples. The mean energy of each
        # symbol is the mean energy per sample multiplied by the number of
        # samples per symbol.
        symbol_energy = signal_power(symbols) * self.samples_per_symbol

        # Due to pulse shaping, we can no longer assume that each symbol has
        # unit energy.
        n0 = symbol_energy / self.es_n0

        # normal() takes the standard deviation.
        noise_r = self.rng.normal(0, np.sqrt(n0 / 2), size=symbols.size)
        noise_i = self.rng.normal(0, np.sqrt(n0 / 2), size=symbols.size)
        return symbols + noise_r + noise_i * 1j

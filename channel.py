from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Planck, speed_of_light

from utils import Component, mean_sample_energy, sample_energies


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
        symbol_energy = mean_sample_energy(symbols) * self.samples_per_symbol

        # Due to pulse shaping, we can no longer assume that each symbol has
        # unit energy.
        n0 = symbol_energy / self.es_n0

        # normal() takes the standard deviation.
        noise_r = self.rng.normal(0, np.sqrt(n0 / 2), size=symbols.size)
        noise_i = self.rng.normal(0, np.sqrt(n0 / 2), size=symbols.size)
        return symbols + noise_r + noise_i * 1j


class ShotNoise(Channel):
    def __init__(self, sampling_rate: float, rx_power_dbm: float) -> None:
        super().__init__()

        assert sampling_rate > 0
        self.sampling_rate = sampling_rate

        rx_power = 10 ** (rx_power_dbm / 10 - 3)
        self.mean_photons_per_sample: float = (
            self.WAVELENGTH * rx_power / (Planck * speed_of_light * sampling_rate)
        )

        self.rng = np.random.default_rng()

        print(self.mean_photons_per_sample)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        super().__call__(symbols)

        mean_energy = mean_sample_energy(symbols)
        photons_per_energy = self.mean_photons_per_sample / mean_energy

        photons = photons_per_energy * symbols

        # FIXME sign never flips, so BPSK/QPSK is immune. Is this the case?
        def noisy(p: NDArray[np.float64]) -> NDArray[np.float64]:
            energy = self.rng.poisson(np.abs(p)) / photons_per_energy
            return np.sqrt(energy) * np.sign(p)

        return noisy(np.real(photons)) + 1j * noisy(np.imag(photons))

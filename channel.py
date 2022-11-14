from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.constants import speed_of_light

from utils import Component, signal_power, samples_squared, power_dbm_to_lin


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


class GNChannel(Channel):
    alpha = 0.046  # Np/km
    gamma = 1.31  # W/km
    l_eff_a = 1 / alpha  # km

    # B_wdm = 50e9  # FIXME
    # B_ref = 50e9  # FIXME

    B_wdm = 4e12  # FIXME
    B_ref = 12.5e9  # FIXME

    def __init__(
        self, length: float, symbol_rate: float, samples_per_symbol: int
    ) -> None:
        super().__init__()

        assert length > 0
        length_km = length / 1000
        self.l_eff = (1 - np.exp(-self.alpha * length_km)) / self.alpha
        assert self.l_eff < self.l_eff_a

        # FIXME
        self.l_eff = 21
        self.l_eff_a = 21

        assert symbol_rate > 0
        assert samples_per_symbol > 0
        self.samples_per_symbol = samples_per_symbol

        beta_2 = (
            -self.WAVELENGTH**2
            * self.GROUP_VELOCITY_DISPERSION
            / (2 * np.pi * speed_of_light)
        )

        eta = 8 / 27 * self.gamma**2 * self.l_eff**2
        eta *= np.arcsinh(np.pi**2 / 2 * beta_2 * self.l_eff_a * self.B_wdm**2)
        eta /= np.pi * beta_2 * self.l_eff_a
        eta *= self.B_ref / symbol_rate**3
        self.eta = eta

        self.rng = np.random.default_rng()

        # print(10 * np.log10(self.eta))
        # FIXME
        self.eta = 10 ** (22 / 10)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Power at each sample.
        p_tx = samples_squared(symbols)

        # FIXME
        p_ase = power_dbm_to_lin(-10)

        # Also per sample.
        osnr = p_tx / (p_ase + self.eta * p_tx**3)
        print("OSNR db", 10 ** np.log10(np.mean(osnr)))
        assert np.all(osnr > 0)

        # FIXME conversion.
        snr = osnr

        stdev = np.sqrt(p_tx * self.samples_per_symbol / (2 * snr))

        noise_r = self.rng.normal(0, stdev, size=symbols.size)
        noise_i = self.rng.normal(0, stdev, size=symbols.size)
        return symbols + noise_r + noise_i * 1j

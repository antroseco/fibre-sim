from abc import abstractmethod
from functools import cache

import numpy as np
from numpy.typing import NDArray

from utils import Component, samples_squared, signal_power


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


class SSFChannel(Channel):
    h = 1000  # m

    def __init__(self, fibre_length: int, sampling_rate: int) -> None:
        super().__init__()

        assert fibre_length > 0
        self.length = fibre_length

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

    @classmethod
    @cache
    def get_linear_arg(cls, size: int, sampling_interval: float) -> NDArray[np.float64]:
        # This is the baseband representation of the signal, which has the same
        # bandwidth as the upconverted PAM signal. It's already centered around
        # 0, so there's no need to subtract the carrier frequency from its
        # spectrum.
        # TODO unify with ChromaticDispersion implementation.
        Df = np.fft.fftfreq(size, sampling_interval)
        # FIXME sign convention.
        return -4j * np.pi**2 * cls.BETA_2 * Df**2 + cls.ATTENUATION

    def split_step_impl(
        self, symbols: NDArray[np.cdouble], step_size: int
    ) -> NDArray[np.cdouble]:
        # Nonlinear term.
        nonlinear_term = np.exp(
            (-1j * step_size * self.NONLINEAR_PARAMETER) * samples_squared(symbols)
        )

        # Linear term.
        linear_arg = self.get_linear_arg(symbols.size, self.sampling_interval)
        linear_term = np.exp(linear_arg * (-step_size / 2))

        # TODO investigate symmetrized schemes.
        return np.fft.ifft(np.fft.fft(symbols * nonlinear_term) * linear_term)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        super().__call__(symbols)
        # FIXME assert is_power_of_2(symbols.size)

        remaining_length = self.length

        # TODO investigate step size h.
        while remaining_length >= self.h:
            symbols = self.split_step_impl(symbols, self.h)
            remaining_length -= self.h

        if remaining_length:
            symbols = self.split_step_impl(symbols, remaining_length)

        return symbols

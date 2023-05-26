from functools import cache
from typing import Optional, Type

import numpy as np
from numpy.typing import NDArray

from utils import (
    Component,
    Signal,
    energy_db_to_lin,
    has_one_polarization,
    has_two_polarizations,
    has_up_to_two_polarizations,
    is_power_of_2,
    power_dbm_to_lin,
    row_size,
    samples_squared,
    signal_power,
)


class Channel(Component):
    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.OPTICAL, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.OPTICAL, np.cdouble, None


class AWGN(Channel):
    def __init__(self, es_n0: float, samples_per_symbol: int) -> None:
        super().__init__()

        assert es_n0 > 0
        self.es_n0 = es_n0

        assert samples_per_symbol > 0
        self.samples_per_symbol = samples_per_symbol

        self.rng = np.random.default_rng()

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        # AWGN is applied at the receiver.
        return Signal.SYMBOLS, np.cdouble, self.samples_per_symbol

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        # AWGN is applied at the receiver.
        return Signal.SYMBOLS, np.cdouble, self.samples_per_symbol

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(symbols)

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

    def __init__(self, fibre_length: int, sampling_rate: float) -> None:
        super().__init__()

        assert fibre_length > 0
        self.length = fibre_length

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

    @staticmethod
    @cache
    def get_linear_term(
        size: int,
        sampling_interval: float,
        beta_2: float,
        attenuation: float,
        step_size: int,
    ) -> NDArray[np.cdouble]:
        # This is the baseband representation of the signal, which has the same
        # bandwidth as the upconverted PAM signal. It's already centered around
        # 0, so there's no need to subtract the carrier frequency from its
        # spectrum.
        # TODO unify with ChromaticDispersion implementation.
        Df = np.fft.fftfreq(size, sampling_interval)
        return np.exp(
            (-step_size / 2) * (4j * np.pi**2 * beta_2 * Df**2 + attenuation)
        )

    def split_step_impl(
        self, symbols: NDArray[np.cdouble], step_size: int
    ) -> NDArray[np.cdouble]:
        # Extra factor in the Manakov equation.
        factor = 8 / 9 if has_two_polarizations(symbols) else 1

        # Nonlinear term.
        nonlinear_term = np.exp(
            (-1j * step_size * factor * self.NONLINEAR_PARAMETER)
            * samples_squared(symbols)
        )

        # Linear term.
        linear_term = self.get_linear_term(
            row_size(symbols),
            self.sampling_interval,
            self.BETA_2,
            self.ATTENUATION,
            step_size,
        )

        return np.fft.ifft(np.fft.fft(symbols * nonlinear_term) * linear_term)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Equations are mostly the same for one and two polarizations. fft and
        # ifft can handle two rows just fine.
        assert has_up_to_two_polarizations(symbols)
        assert is_power_of_2(row_size(symbols))

        remaining_length = self.length

        # TODO investigate step size h.
        while remaining_length >= self.h:
            symbols = self.split_step_impl(symbols, self.h)
            remaining_length -= self.h

        if remaining_length:
            symbols = self.split_step_impl(symbols, remaining_length)

        return symbols


class Splitter(Channel):
    def __init__(self, ratio: int) -> None:
        super().__init__()

        assert ratio > 1 and is_power_of_2(ratio)

        # 3.5 dB loss per coupler (0.5 dB overhead).
        attenuation_dB = 3.5 * np.log2(ratio)
        attenuation = energy_db_to_lin(attenuation_dB)

        self.factor: float = 1 / np.sqrt(attenuation)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_up_to_two_polarizations(symbols)

        return symbols * self.factor


class PolarizationRotation(Channel):
    def __init__(self, angle: Optional[float] = None) -> None:
        super().__init__()

        if angle is None:
            angle = np.random.uniform(-np.pi, np.pi)

        # 2D rotation matrix.
        self.matrix = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_two_polarizations(symbols)

        return self.matrix @ symbols


class DropPolarization(Channel):
    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_two_polarizations(symbols)

        return symbols[0]


class SetPower(Channel):
    def __init__(self, power_dBm: float) -> None:
        super().__init__()

        self.target_power = power_dbm_to_lin(power_dBm)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_up_to_two_polarizations(symbols)

        input_power = signal_power(symbols)
        ratio = np.sqrt(self.target_power / input_power)

        return symbols * ratio

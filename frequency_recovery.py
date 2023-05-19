from functools import cache
from typing import Literal, Optional, Type

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from utils import (
    Component,
    Signal,
    first_polarization,
    has_up_to_two_polarizations,
    is_power_of_2,
)


class FrequencyRecoveryFFT(Component):
    def __init__(
        self,
        symbol_rate: float,
        samples_per_symbol: int,
        fft_size: int,
        window_function: Literal["gaussian", "nuttall"] = "nuttall",
    ) -> None:
        super().__init__()

        assert symbol_rate > 0
        self.symbol_interval = 1 / symbol_rate

        assert samples_per_symbol > 0
        self.samples_per_symbol = samples_per_symbol

        assert is_power_of_2(fft_size)
        self.fft_size = fft_size

        assert window_function in ("gaussian", "nuttall")
        self.window_function = window_function

        self.freq_estimate: Optional[float] = None

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    @staticmethod
    @cache
    def gaussian_window(length: int) -> NDArray[np.float64]:
        # Improving FFT Frequency Measurement Resolution by Parabolic and
        # Gaussian Spectrum Interpolation (M. Gasior and J. L. Gonzalez, 2004)
        # suggests r = 8 gives the best results (Table 2), where r = L/σ.
        return signal.get_window(("gaussian", length / 8), length)

    @staticmethod
    @cache
    def nuttall_4t5(length: int) -> NDArray[np.float64]:
        # Improving FFT Frequency Measurement Resolution by Parabolic and
        # Gaussian Spectrum Interpolation (M. Gasior and J. L. Gonzalez, 2004)
        # gives the coefficients (Table 1) for a Nuttall-like window, with 4
        # terms and a continuous fifth derivate. This has the best performance
        # (Table 2).
        a0 = 10 / 32
        a1 = 15 / 32
        a2 = 6 / 32
        a3 = 1 / 32

        ns = 2 * np.pi * np.arange(length) / length

        return a0 - a1 * np.cos(ns) + a2 * np.cos(2 * ns) - a3 * np.cos(3 * ns)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_up_to_two_polarizations(symbols)

        # Frequency offset should be very similar for both polarizations, so
        # only estimate it using the first polarization.
        symbols = first_polarization(symbols)

        # Resample quickly to 1 sample per symbol. This gives us 12.5 GHz of
        # range at 50 GSa/s, which is more than enough. Downsampling further
        # would result in aliasing (25 GHz of bandwidth), downsampling less
        # would increase the distance (in Hz) between adjacent FFT bins.
        downsampled = symbols[:: self.samples_per_symbol]

        # Avoid edge effects if possible. Slightly crude but works.
        sample = (
            downsampled[: self.fft_size]
            if downsampled.size < 1024 + self.fft_size
            else downsampled[1024 : 1024 + self.fft_size]
        )

        assert self.window_function in ("gaussian", "nuttall")
        window = (
            self.gaussian_window(sample.size)
            if self.window_function == "gaussian"
            else self.nuttall_4t5(sample.size)
        )

        spectrum = np.abs(np.fft.fftshift(np.fft.fft(sample**4 * window)))

        k = np.argmax(spectrum)
        a = spectrum[k - 1]
        b = spectrum[k]
        c = spectrum[k + 1] if k + 1 < spectrum.size else spectrum[k] * 0.9

        # Quadratic interpolation (no logs necessary!).
        p = 0.5 * (a - c) / (a - 2 * b + c)
        assert np.abs(p) <= 0.5

        # We raised the symbols to the 4th power, which multiplied all
        # frequencies by 4.
        self.freq_estimate = (k + p - sample.size // 2) / (
            4 * sample.size * self.symbol_interval
        )

        return symbols

from functools import cache
from typing import Literal, Optional, Type

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from utils import (
    Component,
    Signal,
    first_polarization,
    has_one_polarization,
    has_up_to_two_polarizations,
    row_size,
)


class FrequencyRecovery(Component):
    def __init__(
        self,
        sampling_rate: float,
        window_function: Literal["gaussian", "nuttall"] = "nuttall",
    ) -> None:
        super().__init__()

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

        self.freq_estimate: Optional[float] = None
        self.window_function = window_function

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

    def estimate(self, symbols: NDArray[np.cdouble], fft_size: int) -> None:
        assert has_one_polarization(symbols)

        # FIXME
        sample = symbols[1024 : 1024 + fft_size]

        assert self.window_function in ["gaussian", "nuttall"]
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
            4 * sample.size * self.sampling_interval
        )

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_up_to_two_polarizations(symbols)

        if self.freq_estimate is None:
            # Frequency offset should be very similar for both polarizations, so
            # only estimate it using the first polarization.
            self.estimate(first_polarization(symbols), 1024)

        # Help out the type checker.
        assert self.freq_estimate is not None

        ks = np.arange(row_size(symbols))

        return symbols * np.exp(
            (-2j * np.pi * self.freq_estimate * self.sampling_interval) * ks
        )

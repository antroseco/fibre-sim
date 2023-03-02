from functools import cache
from typing import Optional, Type

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
    def __init__(self, sampling_rate: float) -> None:
        super().__init__()

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

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

    def estimate(self, symbols: NDArray[np.cdouble]) -> None:
        assert has_one_polarization(symbols)

        # Downsample by 2. This is generally safe, as we are typically
        # oversampling by 2 and have low-pass-filtered the data.
        sample = symbols[:256:2]

        window = self.gaussian_window(sample.size)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(sample**4 * window)))

        k = np.argmax(spectrum)
        a = spectrum[k - 1]
        b = spectrum[k]
        c = spectrum[k + 1]

        # Quadratic interpolation (no logs necessary!).
        p = 0.5 * (a - c) / (a - 2 * b + c)
        assert np.abs(p) <= 0.5

        # We raised the symbols to the 4th power, which multiplied all
        # frequencies by 4. We have also downsampled by a factor of 2, so we
        # need to double the sampling interval.
        self.freq_estimate = (k + p - sample.size // 2) / (
            4 * sample.size * (2 * self.sampling_interval)
        )

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_up_to_two_polarizations(symbols)

        if self.freq_estimate is None:
            # Frequency offset should be very similar for both polarizations, so
            # only estimate it using the first polarization.
            self.estimate(first_polarization(symbols))

        # Help out the type checker.
        assert self.freq_estimate is not None

        ks = np.arange(row_size(symbols))

        return symbols * np.exp(
            (-2j * np.pi * self.freq_estimate * self.sampling_interval) * ks
        )

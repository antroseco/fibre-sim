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

    def estimate(self, symbols: NDArray[np.cdouble]) -> None:
        assert has_one_polarization(symbols)

        window = signal.windows.hann(symbols.size, False)

        spectrum = np.abs(np.fft.fft((symbols * window) ** 4))
        freqs = np.fft.fftfreq(symbols.size, self.sampling_interval)

        self.freq_estimate = freqs[np.argmax(spectrum)] / 4

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

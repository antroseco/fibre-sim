from functools import cached_property
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.constants import speed_of_light

from utils import Component, overlap_save


class PulseFilter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    # A span of 32 is quite long for high values of beta (approaching 1),
    # but it's way too short for smaller betas. 128 would be a more
    # appropriate value for betas approaching 0.
    SPAN = 32
    BETA = 0.99

    def __init__(
        self,
        samples_per_symbol: int,
        *,
        up: Optional[int] = None,
        down: Optional[int] = None
    ) -> None:
        super().__init__()

        # Need at least 2 samples per symbol to satisfy the Nyquist–Shannon
        # sampling theorem (raised cosine filter has bandwidth between 1/2T and
        # 1/T depending on the value of beta). This is the one-sided baseband
        # bandwidth of the pulse; even though the negative frequencies contain
        # useful information (as the signal is complex) the Nyquist frequency is
        # the same.
        assert samples_per_symbol > 1
        self.samples_per_symbol = samples_per_symbol

        # It doesn't make sense to set both.
        if up is not None:
            assert up > 0
            assert down is None
        if down is not None:
            assert down > 0
            assert up is None

        self.up = up
        self.down = down

    @cached_property
    def impulse_response(self) -> NDArray[np.float64]:
        rrc = root_raised_cosine(self.BETA, self.samples_per_symbol, self.SPAN)
        assert np.argmax(rrc) == rrc.size // 2
        assert np.isclose(np.sum(np.abs(rrc**2)), 1)

        return rrc

    def upsample(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert self.up is not None
        assert symbols.size > 0

        # This is quite fast actually (https://stackoverflow.com/a/73994667).
        upsampled = np.zeros(symbols.size * self.up - (self.up - 1), dtype=np.cdouble)
        upsampled[:: self.up] = symbols

        return upsampled

    def downsample(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert self.down is not None
        assert symbols.size > 0
        assert symbols.size % self.down == 0

        # Don't bother copying to a new array. Also don't bother running it
        # through a low-pass filter, as the actual ADC wouldn't know about the
        # rest of the samples.
        downsampled = symbols[:: self.down]

        original_energy = np.sum(np.abs(symbols) ** 2)
        downsampled_energy = np.sum(np.abs(downsampled) ** 2)

        # Preserve signal energy. This is crucial if we want the matched filter
        # to work without any external normalization.
        return downsampled * np.sqrt(original_energy / downsampled_energy)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert symbols.ndim == 1

        # Perform any {up, down}sampling first.
        if self.up:
            symbols = self.upsample(symbols)
        if self.down:
            symbols = self.downsample(symbols)

        # Filter the data with the impulse response of the filter.
        filtered = overlap_save(self.impulse_response, symbols, full=True)

        if self.down:
            # FIXME document this.
            return filtered[
                self.SPAN
                * self.samples_per_symbol : -(self.SPAN - 1)
                * self.samples_per_symbol
                + 1
            ]

        # Transmit the symbols as is.
        return filtered


def root_raised_cosine(
    beta: float, samples_per_symbol: int, span: int
) -> NDArray[np.float64]:
    assert 0 < beta < 1
    assert samples_per_symbol > 0
    assert span % 2 == 0

    # Normalize by samples_per_symbol to get time in terms of t/T
    # (T = samples_per_symbol)
    t = np.linspace(-span // 2, span // 2, samples_per_symbol * span, endpoint=False)

    assert t.size % 2 == 0
    assert t.size == samples_per_symbol * span
    assert t[samples_per_symbol * span // 2] == 0

    cos_term = np.cos((1 + beta) * np.pi * t)

    # numpy implements the normalized sinc function, so we need to divide by π
    # to obtain the unnormalized sinc(x) = sin(x)/x.
    sinc_term = np.sinc((1 - beta) * t)
    sinc_term *= (1 - beta) * np.pi / (4 * beta)

    denominator = 1 - (4 * beta * t) ** 2

    p = (cos_term + sinc_term) / denominator
    p *= 4 * beta / (np.pi * np.sqrt(samples_per_symbol))

    # FIXME have to compute the limits when |t/T| = 1/4β.
    assert np.all(np.isfinite(p))

    # Normalize energy. The equation we use does result in a unit energy signal,
    # but only if the span is infinite. Since we truncate the filter, we need to
    # re-normalize the remaining terms.
    p /= np.sqrt(np.sum(p**2))

    return p


class ChromaticDispersion(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    # D = 17 ps/nm/km at λ = 1550 nm according to Digital Coherent Optical
    # Systems.
    GROUP_VELOCITY_DISPERSION = 17 * 1e-12 / (1e-9 * 1e3)

    # Carrier wavelength = 1550 nm.
    WAVELENGTH = 1550e-9

    def __init__(self, length: float, f_c: float) -> None:
        super().__init__()

        assert length > 0
        self.length = length

        assert f_c > 0
        self.sampling_interval = 1 / f_c

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert symbols.ndim == 1

        # This is the baseband representation of the signal, which has the same
        # bandwidth as the upconverted PAM signal. It's already centered around
        # 0, so there's no need to subtract the carrier frequency from its
        # spectrum.
        Df = np.fft.fftfreq(symbols.size, self.sampling_interval)

        cd = np.exp(
            1j
            * np.pi
            * self.WAVELENGTH**2
            * self.GROUP_VELOCITY_DISPERSION
            * self.length
            / speed_of_light
            * Df**2
        )

        # FIXME this is circular convolution.
        return np.fft.ifft(np.fft.fft(symbols) * cd)


class Downsample(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __init__(self, factor: int) -> None:
        super().__init__()

        assert factor > 1
        self.factor = factor

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # FIXME should we normalize the energy here?
        return symbols[:: self.factor]

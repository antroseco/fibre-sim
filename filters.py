from functools import cache

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from utils import Component


class Upsampler(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __init__(self, factor: int) -> None:
        super().__init__()

        assert factor > 1
        self.factor = factor

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert data.ndim == 1

        upsampled = np.zeros(data.size * self.factor, dtype=np.cdouble)
        upsampled[:: self.factor] = data

        return upsampled


class Downsampler(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __init__(self, factor: int, samples_per_symbol: int) -> None:
        super().__init__()

        assert factor > 1
        # FIXME is 1 sample per symbol valid?
        assert samples_per_symbol > 0

        self.factor = factor
        self.samples_per_symbol = samples_per_symbol

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert data.ndim == 1
        assert data.size % self.factor == 0

        # FIXME doesn't work very well for data lengths less than the pulse
        # filter's length

        downsampled = np.zeros(data.size // self.factor, dtype=np.cdouble)
        downsampled = data[:: self.factor]

        return downsampled


class PulseFilter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    # A span of 32 is quite long for high values of beta (approaching 1),
    # but it's way too short for smaller betas. 128 would be a more
    # appropriate value for betas approaching 0.
    SPAN = 32
    BETA = 0.99

    def __init__(self, samples_per_symbol: int) -> None:
        super().__init__()

        assert samples_per_symbol > 0
        self.samples_per_symbol = samples_per_symbol

    @classmethod
    @cache
    def get_frequency_response(
        cls, samples_per_symbol: int, size: int
    ) -> NDArray[np.cdouble]:
        impulse_response = root_raised_cosine(cls.BETA, samples_per_symbol, cls.SPAN)

        # If size is smaller than the length of the impulse response, the
        # impulse response will be cropped.
        assert size >= impulse_response.size

        return np.fft.fft(impulse_response, size)

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Perform a circular convolution using the DFT. We can exploit the
        # circular property to avoid any edge effects without having to store
        # anything from the previous chunk. As the data is random anyway, the
        # data from the current edge is as good as the data from the previous
        # chunk.
        return np.fft.ifft(
            np.fft.fft(data)
            * self.get_frequency_response(self.samples_per_symbol, data.size)
        )


def root_raised_cosine(
    beta: float, samples_per_symbol: int, span: int
) -> NDArray[np.float64]:
    assert 0 < beta < 1
    assert samples_per_symbol % 2 == 0
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

    return p


if __name__ == "__main__":
    rrc = root_raised_cosine(0.11, 128, 32)
    plt.plot(rrc)
    plt.show()
    plt.magnitude_spectrum(rrc.tolist())
    plt.show()

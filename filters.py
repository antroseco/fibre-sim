from functools import cached_property
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.signal import upfirdn

from utils import Component


class PulseFilter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    # A span of 32 is quite long for high values of beta (approaching 1),
    # but it's way too short for smaller betas. 128 would be a more
    # appropriate value for betas approaching 0.
    SPAN = 32
    BETA = 0.99

    def __init__(self, *, up: int = 1, down: int = 1) -> None:
        super().__init__()

        assert (up > 1 and down == 1) or (up == 1 and down > 1)
        self.up = up
        self.down = down

    @cached_property
    def impulse_response(self) -> NDArray[np.float64]:
        return root_raised_cosine(self.BETA, max(self.up, self.down), self.SPAN)

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Perform a circular convolution using the DFT. We can exploit the
        # circular property to avoid any edge effects without having to store
        # anything from the previous chunk. As the data is random anyway, the
        # data from the current edge is as good as the data from the previous
        # chunk.
        filtered = upfirdn(self.impulse_response, data, self.up, self.down)

        if self.down > self.up:
            # The final symbol overruns by its total length minus the number of
            # samples per symbol (its alloted space).
            assert filtered.size == ceil(
                (data.size + self.SPAN * self.down - 1) / self.down
            )

            # If we are downsampling, then we need to remove the convolution
            # artifacts on either side of the signal before we return the
            # symbols for further processing. The filter is symmetrical,
            # affecting SPAN / 2 symbols on either side. Since the signal has
            # been filtered twice, the artifacts now take up SPAN symbols in
            # total. Need to add 1 to the end index as it's exclusive.
            return filtered[self.SPAN : -self.SPAN + 1]
        else:
            # The final symbol overruns by its total length minus the number of
            # samples per symbol (its alloted space).
            assert filtered.size == self.up * (data.size + self.SPAN - 1)

            # Transmit the symbols as is.
            return filtered


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

    # Normalize energy. The equation we use does result in a unit energy signal,
    # but only if the span is infinite. Since we truncate the filter, we need to
    # re-normalize the remaining terms.
    p /= np.sqrt(np.sum(p**2))

    return p

from functools import cached_property
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.signal import upfirdn, get_window, resample, filtfilt
from scipy.signal.windows import kaiser
from scipy.constants import speed_of_light
from channel import AWGN

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


class ChromaticDispersion(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    # D = 17 ps/nm/km at λ = 1550 nm according to Digital Coherent Optical
    # Systems.
    GROUP_VELOCITY_DISPERSION = 17 * 1e-12 / (1e-9 * 1e3)

    # Carrier wavelength = 1550 nm.
    WAVELENGTH = 1550e-9

    def __init__(self, length: float, f_c: float, SpS: int) -> None:
        super().__init__()

        assert length > 0
        self.length = length

        assert f_c > 0
        self.sampling_interval = 1 / f_c

        assert SpS > 1
        self.SpS = SpS

    def __call__(
        self, symbols: NDArray[np.cdouble], prefix=True
    ) -> NDArray[np.cdouble]:
        assert symbols.ndim == 1

        if prefix:
            prefix = np.concatenate(
                [
                    np.zeros(24 * self.SpS),
                    root_raised_cosine(0.11, self.SpS, 8),
                    np.zeros(24 * self.SpS),
                ]
            )

            signal = np.concatenate((prefix, symbols))
        else:
            signal = symbols

        # This is the baseband representation of the signal, which has the same
        # bandwidth as the upconverted PAM signal. It's already centered around
        # 0, so there's no need to subtract the carrier frequency from its
        # spectrum.
        Df = np.fft.fftfreq(signal.size, self.sampling_interval)

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
        return np.fft.ifft(np.fft.fft(signal) * cd)


class CDCompensator(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __init__(self, SpS: int) -> None:
        super().__init__()

        assert SpS >= 1
        self.SpS = SpS

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert symbols.ndim == 1

        prefix = symbols[: 56 * self.SpS]

        # FIXME determine Kaiser window parameter.
        spectrum = np.fft.fft(prefix * kaiser(prefix.size, 4))
        inverse_spectrum = np.exp(-1j * np.angle(spectrum))

        fir = np.fft.ifft(inverse_spectrum)
        fir /= np.sum(np.abs(fir) ** 2)

        return np.convolve(symbols[56 * self.SpS :], fir, mode="same")


def main():
    SpS = 64
    CD = ChromaticDispersion(70e3, SpS * 50e9, SpS)
    noise = AWGN(5e-4)

    prefix = np.concatenate(
        [np.zeros(24 * SpS), root_raised_cosine(0.11, SpS, 8), np.zeros(24 * SpS)]
    )
    signal = np.concatenate([prefix, np.tile(root_raised_cosine(0.11, SpS, 8), 8)])
    rcv_prefix = noise(CD(signal.astype(np.cdouble), prefix=False)[: 56 * SpS])
    plt.plot(signal)
    plt.plot(np.abs(rcv_prefix))
    plt.show()

    # FIXME determine Kaiser window parameter.
    spectrum = np.fft.fft(rcv_prefix * kaiser(rcv_prefix.size, 4))
    inverse_spectrum = np.exp(-1j * np.angle(spectrum))
    plt.plot(np.angle(spectrum))
    plt.plot(np.angle(inverse_spectrum))
    plt.title("Spectrum")
    plt.show()

    fir = np.fft.ifft(inverse_spectrum)
    fir /= np.sum(np.abs(fir) ** 2)
    plt.stem(np.abs(fir))
    plt.title("FIR")
    plt.show()

    test_sig = np.tile(root_raised_cosine(0.11, SpS, 8), 8)
    cd = CD(test_sig.astype(np.cdouble), prefix=False)
    plt.plot(np.real(cd), label="Received (real)")
    plt.plot(np.imag(cd), label="Received (imag)")
    plt.plot(test_sig, label="Original")
    plt.plot(np.real(np.convolve(cd, fir, mode="same")), label="Recovered")
    plt.legend()
    plt.title("Test")
    plt.show()


if __name__ == "__main__":
    main()

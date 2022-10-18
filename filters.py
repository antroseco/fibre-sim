import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from utils import Component


def raised_cosine_spectrum(
    beta: float, points: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    T = 1  # FIXME dummy variable.
    freqs: NDArray = np.linspace(0, 1 / T, points)

    a = (1 - beta) / (2 * T)
    b = (1 + beta) / (2 * T)

    spectrum = np.zeros_like(freqs)

    cond = (np.abs(freqs) > a) & (np.abs(freqs) <= b)
    spectrum[cond] = 0.5 * (1 + np.cos(np.pi * T / beta * (np.abs(freqs[cond]) - a)))
    spectrum[np.abs(freqs) <= a] = 1

    return freqs, spectrum


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
        # assert data.size % self.factor == 0

        # FIXME doesn't work very well for data lengths less than the pulse
        # filter's length

        downsampled = np.zeros(data.size // self.factor, dtype=np.cdouble)
        downsampled = data[:: self.factor]

        return downsampled


class PulseFilter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __init__(self, samples_per_symbol: int, span: int) -> None:
        super().__init__()
        assert samples_per_symbol > 0
        assert span % 2 == 0

        self.samples_per_symbol = samples_per_symbol
        self.span = span

        self.impulse_response = root_raised_cosine(0.9, samples_per_symbol, span)

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        return np.convolve(data, self.impulse_response, mode="same")


def root_raised_cosine(
    beta: float, samples_per_symbol: int, span: int
) -> NDArray[np.float64]:
    # samples_per_symbol is samples per symbol (so it's equivalent to T)
    assert span % 2 == 0

    # Normalize by samples_per_symbol to get time in terms of t/T
    t = np.arange(-samples_per_symbol * span // 2, samples_per_symbol * span // 2 + 1)
    # t = np.linspace(-5, 5, 1001, endpoint=True)
    T = samples_per_symbol
    p = np.empty_like(t)
    assert t.size == samples_per_symbol * span + 1

    cos_term = np.cos((1 + beta) * np.pi * t / T)

    sinc_term = np.sinc(((1 - beta) * np.pi * t / T) / (np.pi))
    sinc_term *= (1 - beta) * np.pi / (4 * beta)

    denominator = 1 - (4 * beta * t / T) ** 2

    p = (cos_term + sinc_term) / denominator
    p *= 4 * beta / (np.pi * np.sqrt(T))

    assert np.all(np.isfinite(p))

    return p


if __name__ == "__main__":
    points = 33

    rrc = root_raised_cosine(0.99, 10, 8)
    plt.plot(rrc)
    plt.show()

    # freqs, s = raised_cosine_spectrum(0.25, 16)  # points // 2 + 1)

    # # plt.plot(freqs, s)
    # # plt.show()

    # h = np.fft.irfft(s, points)
    # h = np.concatenate([h[points // 2 + 1 :], h[: points // 2 + 1]])
    # plt.stem(h, label="raised cosine")

    # rh = np.fft.irfft(np.sqrt(s), points)
    # rh = np.concatenate([rh[points // 2 + 1 :], rh[: points // 2 + 1]])
    # # plt.stem(rh, markerfmt="s", label="root raised cosine")

    # plt.legend()
    # plt.show()

    # cc = np.convolve(rh, rh[::-1], mode="same")

    # plt.stem(cc / np.max(cc), label="Convolution results")
    # plt.stem(h / np.max(h), label="Actual")
    # plt.legend()
    # plt.show()

    # signal = np.zeros(9 * 10)
    # signal[::9] = np.random.choice([-1, 1], size=10)

    # plt.stem(np.convolve(signal, h))

    # plt.show()

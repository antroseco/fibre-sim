import numpy as np
from numpy.typing import NDArray

from matplotlib import pyplot as plt


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


if __name__ == "__main__":
    points = 21

    freqs, s = raised_cosine_spectrum(0.5, points // 2 + 1)

    plt.plot(freqs, s)
    plt.show()

    h = np.fft.irfft(s, points)
    h = np.concatenate([h[points // 2 + 1 :], h[: points // 2 + 1]])
    plt.stem(h, label="raised cosine")

    rh = np.fft.irfft(np.sqrt(s), points)
    rh = np.concatenate([rh[points // 2 + 1 :], rh[: points // 2 + 1]])
    plt.stem(rh, markerfmt="s", label="root raised cosine")

    plt.legend()
    plt.show()

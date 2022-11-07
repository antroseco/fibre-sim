from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.special import erfc


class Component(ABC):
    input_type = None
    output_type = None

    @abstractmethod
    def __call__(self, data: NDArray) -> NDArray:
        pass


class Plotter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        _, ax = plt.subplots()
        ax.stem(np.real(data[:64]), markerfmt="bo", label="In-phase")
        ax.stem(np.imag(data[:64]), markerfmt="go", label="Quadrature")
        ax.legend()
        plt.show()
        # FIXME this shows all figures...
        # FIXME need to close the figure after we're done with it.
        return data


class SpectrumPlotter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        _, ax = plt.subplots()
        ax.magnitude_spectrum(symbols.tolist())
        plt.show()

        return symbols


def calculate_awgn_ber_with_bpsk(eb_n0: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * erfc(np.sqrt(eb_n0))


def calculate_awgn_ser_with_qam(
    M: int, eb_n0: NDArray[np.float64]
) -> NDArray[np.float64]:
    es_n0 = 4 * eb_n0

    # Equation 2.16 in Digital Coherent Optical Systems.
    return 2 * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * es_n0 / (2 * (M - 1))))


def calculate_awgn_ber_with_16qam(eb_n0: NDArray[np.float64]) -> NDArray[np.float64]:
    # This is the SER. Divide by 4 (bits per symbol) to get the approximate BER.
    return calculate_awgn_ser_with_qam(16, eb_n0) / 4


def next_power_of_2(value: int) -> int:
    # Bit-twiddling trick from Hacker's Delight by Henry S. Warren.
    # Relies on 32-bit unsigned integer math.
    assert value > 0

    x = np.uint32(value - 1)

    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16

    return int(x + 1)


def is_power_of_2(value: int) -> bool:
    # Well-known trick to check if a number is a power of 2.
    return value > 0 and value & (value - 1) == 0


def overlap_save(h: NDArray, x: NDArray, full: bool = False) -> NDArray[np.cdouble]:
    # Ensure neither array is empty.
    assert h.ndim == 1
    assert h.size >= 1
    assert x.ndim == 1
    assert x.size >= 1

    # N is the frame length. Based on:
    # https://commons.wikimedia.org/wiki/File:FFT_size_vs_filter_length_for_Overlap-add_convolution.svg
    N = max(128, next_power_of_2(h.size) * 8)

    # M is the order of the filter.
    M = h.size - 1
    assert M < N

    # The number of useful data points per frame.
    step_size = N - M

    # Compute the DFT of h, but append enough zeros to match the frame length N.
    H = np.fft.fft(h, N)

    # Pad x to capture the complete convolution of the two signals (equivalent
    # to numpy mode "full"). The default is equivalent to "same".
    if full:
        x = np.pad(x, (0, M))

    # Output array.
    y = np.zeros_like(x, dtype=np.cdouble)

    # Overlap cache; initially zero.
    last_m = np.zeros(M)

    for i in range(0, x.size, step_size):
        assert last_m.size == M

        data_len = min(step_size, x.size - i)

        # np.concatenate would had made a copy anyway. This approach copies too,
        # but also zero pads the array to length N for free.
        frame = np.zeros(N, dtype=np.cdouble)
        frame[:M] = last_m
        frame[M : M + data_len] = x[i : i + data_len]

        X = np.fft.fft(frame)
        yt = np.fft.ifft(H * X)

        y[i : i + data_len] = yt[M : M + data_len]

        # If the filter order is not zero, we need to save enough data to avoid
        # edge effects in the next frame.
        if M:
            # A view, so no copy.
            last_m = frame[-M:]

    return y


def signal_energy(signal: NDArray) -> float:
    return np.sum(np.real(np.conj(signal) * signal))


def mean_sample_energy(signal: NDArray) -> float:
    return np.mean(np.real(np.conj(signal) * signal))

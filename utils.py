from abc import ABC, abstractmethod
from typing import overload

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


class PlotSignal(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __call__(self, data: NDArray) -> NDArray:
        assert data.ndim == 1
        assert data.size > 0

        s_real = np.real(data[:1024])
        s_imag = np.imag(data[:1024])

        _, axs = plt.subplots(nrows=2, ncols=3)

        # Constellation diagram.
        ax = axs[0][0]
        ax.scatter(s_real, s_imag)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.axhline(color="black")
        ax.axvline(color="black")

        # Unused plot.
        axs[1][0].set_axis_off()

        # Signal (real component).
        ax = axs[0][1]
        ax.stem(s_real)
        ax.set_xlim(-4, 64)
        ax.set_xlabel("Sample")
        ax.set_ylabel("In-phase")

        # Signal (imaginary component).
        ax = axs[1][1]
        ax.stem(s_imag)
        ax.set_xlim(-4, 64)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Quadrature")

        # Spectrum.
        axs[0][2].magnitude_spectrum(data.tolist(), sides="twosided")
        axs[1][2].phase_spectrum(data.tolist(), sides="twosided")

        plt.show()

        return data


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


def normalize_energy(signal: NDArray) -> NDArray:
    return signal / np.sqrt(signal_energy(signal))


@overload
def energy_db_to_lin(db: float) -> float:
    ...


@overload
def energy_db_to_lin(db: NDArray) -> NDArray[np.float64]:
    ...


def energy_db_to_lin(db: float | NDArray) -> float | NDArray[np.float64]:
    return 10 ** (db / 10)


def power_dbm_to_lin(dbm: float) -> float:
    return 1e-3 * energy_db_to_lin(dbm)

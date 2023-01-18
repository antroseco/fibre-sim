from abc import ABC, abstractmethod
from typing import Callable, overload

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.constants import speed_of_light
from scipy.special import erfc


class Component(ABC):
    input_type = None
    output_type = None

    # Carrier wavelength = 1550 nm.
    WAVELENGTH = 1550e-9

    # α = 0.2 dB/km at λ = 1550 nm according to Digital Coherent Optical
    # Systems.
    ATTENUATION = 0.2 / (1e4 * np.log10(np.e))  # Np/m

    # D = 17 ps/nm/km at λ = 1550 nm according to Digital Coherent Optical
    # Systems.
    GROUP_VELOCITY_DISPERSION = 17 * 1e-12 / (1e-9 * 1e3)  # s/m^2

    BETA_2 = -GROUP_VELOCITY_DISPERSION * WAVELENGTH**2 / (2 * np.pi * speed_of_light)

    # γ = 1.3 /W*km (roughly) according to Digital Coherent Optical Systems.
    # This includes information about the effective area and the nonlinear index
    # of the fibre, as well as the wavelength.
    NONLINEAR_PARAMETER = 1.3e-3  # /W*m

    @abstractmethod
    def __call__(self, data: NDArray) -> NDArray:
        pass


def plot_signal(component: str, signal: NDArray) -> None:
    if signal.ndim == 2:
        plot_signal(component, signal[0])
        plot_signal(component, signal[1])
        return

    assert signal.ndim == 1
    assert signal.size > 0

    s_real = np.real(signal[:1024])
    s_imag = np.imag(signal[:1024])

    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.suptitle(f"After {component}")

    # Constellation diagram.
    ax = axs[0][0]
    ax.scatter(s_real, s_imag, alpha=0.2)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.axhline(color="black")
    ax.axvline(color="black")

    # Unused plot.
    axs[1][0].set_axis_off()

    # Signal (real component).
    ax = axs[0][1]
    ax.stem(s_real)
    ax.set_xlim(-4, 512)
    ax.set_xlabel("Sample")
    ax.set_ylabel("In-phase")

    # Signal (imaginary component).
    ax = axs[1][1]
    ax.stem(s_imag)
    ax.set_xlim(-4, 512)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Quadrature")

    # Spectrum.
    axs[0][2].magnitude_spectrum(signal.tolist(), sides="twosided")
    axs[1][2].phase_spectrum(signal.tolist(), sides="twosided")

    plt.show()


@overload
def calculate_awgn_ber_with_bpsk(eb_n0: float) -> float:
    ...


@overload
def calculate_awgn_ber_with_bpsk(eb_n0: NDArray[np.float64]) -> NDArray[np.float64]:
    ...


def calculate_awgn_ber_with_bpsk(
    eb_n0: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return 0.5 * erfc(np.sqrt(eb_n0))


@overload
def calculate_awgn_ser_with_qam(M: int, eb_n0: float) -> float:
    ...


@overload
def calculate_awgn_ser_with_qam(
    M: int, eb_n0: NDArray[np.float64]
) -> NDArray[np.float64]:
    ...


def calculate_awgn_ser_with_qam(
    M: int, eb_n0: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    es_n0 = 4 * eb_n0

    # Equation 2.16 in Digital Coherent Optical Systems.
    return 2 * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * es_n0 / (2 * (M - 1))))


@overload
def calculate_awgn_ber_with_16qam(eb_n0: float) -> float:
    ...


@overload
def calculate_awgn_ber_with_16qam(eb_n0: NDArray) -> NDArray[np.float64]:
    ...


def calculate_awgn_ber_with_16qam(
    eb_n0: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
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


def samples_squared(signal: NDArray) -> NDArray[np.float64]:
    assert has_up_to_two_polarizations(signal)

    squares = np.real(np.conj(signal) * signal)

    if has_two_polarizations(signal):
        # Sum both polarizations.
        return squares.sum(axis=0)

    return squares


def signal_energy(signal: NDArray, sampling_interval: float = 1) -> float:
    # Energy is proportional to the time between samples.
    return sampling_interval * np.sum(samples_squared(signal))


def signal_power(signal: NDArray) -> float:
    # Power is simply the mean squared sample. Time cancels out.
    return np.mean(samples_squared(signal))


def normalize_energy(signal: NDArray) -> NDArray:
    return signal / np.sqrt(signal_energy(signal))


def normalize_power(signal: NDArray) -> NDArray:
    return signal / np.sqrt(signal_power(signal))


@overload
def energy_db_to_lin(db: float) -> float:
    ...


@overload
def energy_db_to_lin(db: NDArray) -> NDArray[np.float64]:
    ...


def energy_db_to_lin(db: float | NDArray) -> float | NDArray[np.float64]:
    return 10 ** (db / 10)


def power_dbm_to_lin(dbm: float) -> float:
    return energy_db_to_lin(dbm - 30)


def has_one_polarization(array: NDArray) -> bool:
    # Array shouldn't be empty.
    if not array.size:
        return False

    return array.ndim == 1


def has_two_polarizations(array: NDArray) -> bool:
    # Array shouldn't be empty.
    if not array.size:
        return False

    if array.ndim != 2:
        return False

    nrows, _ = array.shape
    return nrows == 2


def has_up_to_two_polarizations(array: NDArray) -> bool:
    return has_one_polarization(array) or has_two_polarizations(array)


def row_size(array: NDArray) -> int:
    assert has_up_to_two_polarizations(array)

    return array.size if has_one_polarization(array) else array.size // 2


def for_each_polarization(
    fn: Callable[[Component, NDArray], NDArray]
) -> Callable[[Component, NDArray], NDArray]:
    def wrapper(self: Component, array: NDArray) -> NDArray:
        # Nothing to do here.
        if has_one_polarization(array):
            return array

        assert has_up_to_two_polarizations(array)

        pol_v = fn(self, array[0])
        pol_h = fn(self, array[1])

        return np.vstack((pol_v, pol_h))

    return wrapper

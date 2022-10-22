from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.special import erfc


class Component(ABC):
    input_type = None
    output_type = None

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass


class Plotter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __call__(self, data: np.ndarray) -> np.ndarray:
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


def calculate_awgn_ber_with_bpsk(eb_n0: np.ndarray):
    return 0.5 * erfc(np.sqrt(eb_n0))


def calculate_awgn_ser_with_qam(
    M: int, eb_n0: NDArray[np.float64]
) -> NDArray[np.float64]:
    es_n0 = 4 * eb_n0

    # Equation 2.16 in Digital Coherent Optical Systems.
    return 2 * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * es_n0 / (2 * (M - 1))))


def calculate_n0(eb_n0: float, bits_per_symbol: int) -> float:
    # Energy per symbol.
    es_n0 = eb_n0 * bits_per_symbol

    # Each symbol has unit energy, so N0 is just the reciprocal.
    return 1 / es_n0


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

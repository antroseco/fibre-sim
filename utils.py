from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc


class Component(ABC):
    input_type = None
    output_type = None

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass


def calculate_awgn_ber_with_bpsk(eb_n0: np.ndarray):
    return 0.5 * erfc(np.sqrt(eb_n0))


def calculate_awgn_ser_with_qam(
    M: int, eb_n0: NDArray[np.float64]
) -> NDArray[np.float64]:
    es_n0 = 4 * eb_n0

    # Equation 2.16 in Digital Coherent Optical Systems.
    return 2 * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * es_n0 / (2 * (M - 1))))

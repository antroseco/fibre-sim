from abc import ABC, abstractmethod
from itertools import cycle
from typing import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes
from scipy.special import erfc


class Component(ABC):
    input_type = None
    output_type = None

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass


def plot_ber(
    ax: Axes,
    eb_n0_db: Iterable[float],
    bers: Sequence[Iterable[float]],
    labels: Sequence[str],
):
    assert len(bers) == len(labels)

    markers = cycle(("o", "x", "s", "*"))

    for ber, label in zip(bers, labels):
        ax.plot(eb_n0_db, ber, marker=next(markers), label=label)

    ax.set_yscale("log")
    ax.set_ylabel("BER")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.legend()


def calculate_awgn_ber_with_bpsk(eb_n0: np.ndarray):
    return 0.5 * erfc(np.sqrt(eb_n0))

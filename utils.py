from itertools import cycle
from typing import Iterable, Sequence

from matplotlib.axes import Axes


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

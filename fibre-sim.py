from itertools import cycle
from typing import Iterable, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.special import erfc


# BPSK over AWGN channel.
def simulate_impl(length: int, N0: float) -> int:
    # Generate random bits.
    rng = np.random.default_rng()
    bits = rng.integers(low=0, high=1, endpoint=True, size=length)

    # Map bits to symbols (1 -> -1, 0 -> 1).
    symbols = 1 - 2 * bits

    # Simulate AWGN. Note that normal() takes the standard deviation.
    rx_samples = symbols + rng.normal(0, np.sqrt(N0 / 2), size=length)

    # Determine received bits.
    rx_bits = rx_samples < 0

    # Calculate number of errors.
    return np.sum(rx_bits ^ bits)


def simulate(len: int, N0: float) -> int:
    # Limit our memory usage.
    MAX_LEN = 10_000_000

    errors = 0
    while len:
        chunk_len = min(len, MAX_LEN)
        errors += simulate_impl(chunk_len, N0)
        len -= chunk_len

    return errors


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


eb_n0_db = np.arange(1, 8, 0.5)
eb_n0 = 10 ** (eb_n0_db / 10)

theoretical_bers = 0.5 * erfc(np.sqrt(eb_n0))
bers = [simulate(10**6, 1 / i) / 10**6 for i in eb_n0]

print(bers)

_, ax = plt.subplots()
plot_ber(ax, eb_n0_db, (theoretical_bers, bers), ("Theoretical", "Simulation"))
plt.show()

from itertools import cycle
from typing import Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from channel import AWGN
from data_stream import PseudoRandomStream
from filters import Downsampler, PulseFilter, Upsampler
from modulation import (
    Demodulator16QAM,
    DemodulatorBPSK,
    DemodulatorQPSK,
    Modulator16QAM,
    ModulatorBPSK,
    ModulatorQPSK,
)
from system import build_system
from utils import (
    Component,
    Plotter,
    calculate_awgn_ber_with_bpsk,
    calculate_awgn_ser_with_qam,
    calculate_n0,
)


def energy_db_to_lin(db):
    return 10 ** (db / 10)


def simulate_impl(system: Sequence[Component], length: int) -> float:
    return build_system(PseudoRandomStream(), system)(length) / length


def simulate_bpsk(length: int, eb_n0: float) -> float:
    # BPSK over AWGN channel.
    N0 = calculate_n0(eb_n0, 1)
    system = (
        ModulatorBPSK(),
        Upsampler(8),
        PulseFilter(8, 4),
        AWGN(N0),
        PulseFilter(8, 4),
        Downsampler(8, 8),
        DemodulatorBPSK(),
    )
    return simulate_impl(system, length)


def simulate_qpsk(length: int, eb_n0: float) -> float:
    # QPSK over AWGN channel.
    N0 = calculate_n0(eb_n0, 2)
    system = (
        ModulatorQPSK(),
        Upsampler(8),
        PulseFilter(8, 4),
        AWGN(N0),
        PulseFilter(8, 4),
        Downsampler(8, 8),
        DemodulatorQPSK(),
    )
    return simulate_impl(system, length)


def simulate_16qam(length: int, eb_n0: float) -> float:
    # 16-QAM over AWGN channel.
    N0 = calculate_n0(eb_n0, 4)
    system = (
        Modulator16QAM(),
        Upsampler(8),
        PulseFilter(8, 4),
        # AWGN(N0),
        PulseFilter(8, 4),
        Downsampler(8, 8),
        Plotter(),
        Demodulator16QAM(),
    )
    return simulate_impl(system, length)


def run_simulation(
    ax: Axes, target_ber: float, simulation: Callable[[int, float], float], **kwargs
) -> None:
    INITIAL_LENGTH = 4 * 10**4
    MAX_LENGTH = 10**7
    MAX_EB_N0_DB = 12

    bers: list[float] = []

    for eb_n0_db in range(1, MAX_EB_N0_DB + 1):
        eb_n0 = energy_db_to_lin(eb_n0_db)
        length = (
            # Magic heuristic that estimates how many samples we need to get a
            # decent BER estimate. Takes care to round the result to the next
            # lowest multiple of 4.
            min(int(4000 / bers[-1]) & ~0b11, MAX_LENGTH)
            if bers
            else INITIAL_LENGTH
        )

        bers.append(simulation(length, eb_n0))

        print(bers[-1])
        if bers[-1] < target_ber:
            break

    ax.plot(range(1, len(bers) + 1), bers, alpha=0.6, **kwargs)


if __name__ == "__main__":
    TARGET_BER = 10**-3

    eb_n0_db = np.linspace(1, 12, 100)
    eb_n0 = energy_db_to_lin(eb_n0_db)

    th_ber_psk = calculate_awgn_ber_with_bpsk(eb_n0)
    # This is the SER. Divide by 4 (bits per symbol) to get the approximate BER.
    th_ber_16qam = calculate_awgn_ser_with_qam(16, eb_n0) / 4

    _, ax = plt.subplots()

    ax.plot(eb_n0_db, th_ber_psk, alpha=0.2, linewidth=5, label="Theoretical BPSK/QPSK")
    ax.plot(eb_n0_db, th_ber_16qam, alpha=0.2, linewidth=5, label="Theoretical 16-QAM")

    markers = cycle(("o", "x", "s", "*"))

    for simulation, label in (
        # (simulate_bpsk, "Simulated BPSK"),
        # (simulate_qpsk, "Simulated QPSK"),
        (simulate_16qam, "Simulated 16-QAM"),
    ):
        run_simulation(ax, TARGET_BER, simulation, label=label, marker=next(markers))

    ax.set_ylim(TARGET_BER / 4)
    ax.set_yscale("log")
    ax.set_ylabel("BER")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.legend()

    plt.show()

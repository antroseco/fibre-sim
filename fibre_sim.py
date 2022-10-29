from concurrent.futures import ProcessPoolExecutor
from itertools import cycle
from typing import Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt

from channel import AWGN
from data_stream import PseudoRandomStream
from filters import CDCompensator, ChromaticDispersion, Downsample, PulseFilter
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
    SpectrumPlotter,
    calculate_awgn_ber_with_bpsk,
    calculate_awgn_ser_with_qam,
    is_power_of_2,
    next_power_of_2,
)

CHANNEL_SPS = 16
RECEIVER_SPS = 2
FIBRE_LENGTH = 100_000  # 100 km
SYMBOL_RATE = 50 * 10**9  # 50 GS/s
TARGET_BER = 10**-3


def energy_db_to_lin(db):
    return 10 ** (db / 10)


def simulate_impl(system: Sequence[Component], length: int) -> float:
    # We take the FFT of the modulated data, so it's best that the data is a
    # power of 2.
    assert is_power_of_2(length)

    return build_system(PseudoRandomStream(), system)(length) / length


def simulate_bpsk(length: int, eb_n0: float) -> float:
    # BPSK over AWGN channel.
    system = (
        ModulatorBPSK(),
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        ChromaticDispersion(FIBRE_LENGTH, SYMBOL_RATE * CHANNEL_SPS),
        AWGN(eb_n0 * ModulatorBPSK.bits_per_symbol, RECEIVER_SPS),
        PulseFilter(RECEIVER_SPS, down=CHANNEL_SPS // RECEIVER_SPS),
        CDCompensator(
            FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, 129
        ),  # FIXME order
        Downsample(RECEIVER_SPS),
        DemodulatorBPSK(),
    )
    return simulate_impl(system, length)


def simulate_qpsk(length: int, eb_n0: float) -> float:
    # QPSK over AWGN channel.
    system = (
        ModulatorQPSK(),
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        ChromaticDispersion(FIBRE_LENGTH, SYMBOL_RATE * CHANNEL_SPS),
        AWGN(eb_n0 * ModulatorQPSK.bits_per_symbol, RECEIVER_SPS),
        PulseFilter(RECEIVER_SPS, down=CHANNEL_SPS // RECEIVER_SPS),
        CDCompensator(
            FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, 129
        ),  # FIXME order
        Downsample(RECEIVER_SPS),
        DemodulatorQPSK(),
    )
    return simulate_impl(system, length)


def simulate_16qam(length: int, eb_n0: float) -> float:
    # 16-QAM over AWGN channel.
    system = (
        Modulator16QAM(),
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        ChromaticDispersion(FIBRE_LENGTH, SYMBOL_RATE * CHANNEL_SPS),
        AWGN(eb_n0 * Modulator16QAM.bits_per_symbol, RECEIVER_SPS),
        PulseFilter(RECEIVER_SPS, down=CHANNEL_SPS // RECEIVER_SPS),
        CDCompensator(
            FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, 251
        ),  # FIXME order
        Downsample(RECEIVER_SPS),
        Demodulator16QAM(),
    )
    return simulate_impl(system, length)


def run_simulation(
    simulation: Callable[[int, float], float]
) -> tuple[list[int], list[float]]:
    INITIAL_LENGTH = 2**14  # 16,384
    MAX_LENGTH = 2**24  # 16,777,216
    MAX_EB_N0_DB = 12

    eb_n0_dbs: list[int] = []
    bers: list[float] = []

    for eb_n0_db in range(1, MAX_EB_N0_DB + 1):
        eb_n0 = energy_db_to_lin(eb_n0_db)
        length = (
            # Magic heuristic that estimates how many samples we need to get a
            # decent BER estimate. Rounds the result to the next greatest power
            # of 2, as we will be taking the FFT of the data later.
            min(next_power_of_2(int(4000 / bers[-1])), MAX_LENGTH)
            if bers
            else INITIAL_LENGTH
        )

        eb_n0_dbs.append(eb_n0_db)
        bers.append(simulation(length, eb_n0))

        if bers[-1] < TARGET_BER:
            break

    return eb_n0_dbs, bers


def main() -> None:
    _, ax = plt.subplots()

    markers = cycle(("o", "x", "s", "*"))
    labels = ("Simulated BPSK", "Simulated QPSK", "Simulated 16-QAM")
    simulations = (simulate_bpsk, simulate_qpsk, simulate_16qam)

    with ProcessPoolExecutor() as executor:
        for label, marker, (eb_n0_dbs, bers) in zip(
            labels,
            markers,
            executor.map(run_simulation, simulations),
        ):
            ax.plot(eb_n0_dbs, bers, alpha=0.6, label=label, marker=marker)

    eb_n0_db = np.linspace(1, 12, 100)
    eb_n0 = energy_db_to_lin(eb_n0_db)

    th_ber_psk = calculate_awgn_ber_with_bpsk(eb_n0)
    # This is the SER. Divide by 4 (bits per symbol) to get the approximate BER.
    th_ber_16qam = calculate_awgn_ser_with_qam(16, eb_n0) / 4

    ax.plot(eb_n0_db, th_ber_psk, alpha=0.2, linewidth=5, label="Theoretical BPSK/QPSK")
    ax.plot(eb_n0_db, th_ber_16qam, alpha=0.2, linewidth=5, label="Theoretical 16-QAM")

    ax.set_ylim(TARGET_BER / 4)
    ax.set_yscale("log")
    ax.set_ylabel("BER")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()

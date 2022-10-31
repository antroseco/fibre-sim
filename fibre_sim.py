from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import cycle
from multiprocessing import cpu_count
from typing import Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

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
    calculate_awgn_ber_with_16qam,
    calculate_awgn_ber_with_bpsk,
    is_power_of_2,
    next_power_of_2,
)

CHANNEL_SPS = 16
RECEIVER_SPS = 2
FIBRE_LENGTH = 100_000  # 100 km
SYMBOL_RATE = 50 * 10**9  # 50 GS/s
TARGET_BER = 0.5 * 10**-3

# Each simulation operates on its own data and the workload is very SIMD/math
# heavy. As the execution unit is the bottleneck, there is no benefit to
# hyper-threading. See https://stackoverflow.com/a/30720868.
#
# Tests on an Intel i7-9750H (6 cores, 12 threads).
#
# With 12 processes:
# Executed in   40.35 secs
#    usr time  194.24 secs
#    sys time    8.16 secs
#
# With 6 processes:
# Executed in   38.28 secs
#    usr time  205.69 secs
#    sys time    2.34 secs
#
# Therefore we should divide by 2 to avoid scheduling two processes on one core.
PHYSICAL_CORES = cpu_count() // 2


def energy_db_to_lin(db: NDArray) -> NDArray[np.float64]:
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
    p_executor: ProcessPoolExecutor,
    ber_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    simulation: Callable[[int, float], float],
) -> tuple[NDArray[np.int64], list[float]]:
    MAX_LENGTH = 2**24  # 16,777,216
    MAX_EB_N0_DB = 12

    eb_n0_dbs = np.arange(1, MAX_EB_N0_DB + 1)
    eb_n0s = energy_db_to_lin(eb_n0_dbs)

    # Only simulate up to the target BER.
    expected_bers = filter(lambda ber: ber > TARGET_BER, ber_function(eb_n0s))

    # Magic heuristic that estimates how many samples we need to get a decent
    # BER estimate. Rounds the result to the next greatest power of 2, as we
    # will be taking the FFT of the data later.
    lengths = [min(next_power_of_2(int(4000 / i)), MAX_LENGTH) for i in expected_bers]

    # TODO would be nice if this returned the iterator and the plot updated as
    # the results came in.
    bers = list(p_executor.map(simulation, lengths, eb_n0s))
    return eb_n0_dbs[: len(bers)], bers


def main() -> None:
    _, ax = plt.subplots()

    markers = cycle(("o", "x", "s", "*"))
    labels = ("Simulated BPSK", "Simulated QPSK", "Simulated 16-QAM")
    simulations = (simulate_bpsk, simulate_qpsk, simulate_16qam)
    ber_estimators = (
        calculate_awgn_ber_with_bpsk,
        calculate_awgn_ber_with_bpsk,
        calculate_awgn_ber_with_16qam,
    )

    with (
        ProcessPoolExecutor(max_workers=PHYSICAL_CORES) as p_executor,
        ThreadPoolExecutor(max_workers=len(simulations)) as t_executor,
    ):
        fn = partial(run_simulation, p_executor)

        for label, marker, (eb_n0_dbs, bers) in zip(
            labels,
            markers,
            t_executor.map(fn, ber_estimators, simulations),
        ):
            ax.plot(eb_n0_dbs, bers, alpha=0.6, label=label, marker=marker)

    eb_n0_db = np.linspace(1, 12, 100)
    eb_n0 = energy_db_to_lin(eb_n0_db)

    th_ber_psk = calculate_awgn_ber_with_bpsk(eb_n0)
    th_ber_16qam = calculate_awgn_ber_with_16qam(eb_n0)

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

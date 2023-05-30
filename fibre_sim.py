from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import cycle
from multiprocessing import cpu_count
from typing import Callable, Optional, Sequence, Type

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray

from channel import (
    AWGN,
    DropPolarization,
    PolarizationRotation,
    SetPower,
    Splitter,
    SSFChannel,
)
from data_stream import PseudoRandomStream
from filters import (
    AdaptiveEqualizer,
    AdaptiveEqualizerAlamouti,
    CDCompensator,
    ChromaticDispersion,
    Decimate,
    PulseFilter,
)
from laser import ContinuousWaveLaser, NoisyLaser
from modulation import (
    AlamoutiEncoder,
    Demodulator,
    Demodulator16QAM,
    DemodulatorBPSK,
    DemodulatorQPSK,
    DPModulator,
    IQModulator,
    Modulator,
    Modulator16QAM,
    ModulatorBPSK,
    ModulatorQPSK,
)
from phase_recovery import DecisionDirected
from receiver import (
    Digital90degHybrid,
    NoisyHeterodyneFrontEnd,
    NoisyOpticalFrontEnd,
    OpticalFrontEnd,
)
from system import build_system
from utils import (
    Component,
    NormalizePower,
    Signal,
    calculate_awgn_ber_with_16qam,
    calculate_awgn_ber_with_bpsk,
    energy_db_to_lin,
    is_power_of_2,
    next_power_of_2,
    plot_filter_with_gd,
)

plt.rcParams.update({"font.size": 14})

CHANNEL_SPS = 16
RECEIVER_SPS = 2
# We only need 17 taps at 25 km.
CDC_TAPS = 17
FIBRE_LENGTH = 25_000  # 25 km
SPLITTING_POINT = 24_000  # 24 km
CONSUMERS = 64
SYMBOL_RATE = 50 * 10**9  # 50 GS/s
TARGET_BER = 0.5 * 10**-3
DDPR_BUFFER_SIZE = 64
LASER_LINEWIDTH_ESTIMATE = 100e3  # 100 kHz
SNR_ESTIMATE = 10  # 10 dB

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


def simulate_impl(system: Sequence[Component], length: int) -> float:
    # We take the FFT of the modulated data, so it's best that the data is a
    # power of 2.
    assert is_power_of_2(length)

    return build_system(PseudoRandomStream(), system)(length)


def awgn_link(es_n0: float) -> Sequence[Component]:
    return (
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        IQModulator(NoisyLaser(10, SYMBOL_RATE * CHANNEL_SPS)),
        ChromaticDispersion(FIBRE_LENGTH, SYMBOL_RATE * CHANNEL_SPS),
        OpticalFrontEnd(),
        Decimate(CHANNEL_SPS // RECEIVER_SPS),
        AWGN(es_n0, RECEIVER_SPS),
        CDCompensator(FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, CDC_TAPS),
        PulseFilter(RECEIVER_SPS, down=RECEIVER_SPS),
    )


def make_awgn_simulation(
    modulator: Type[Modulator],
    demodulator: Type[Demodulator],
    length: int,
    eb_n0: float,
) -> float:
    es_n0 = eb_n0 * modulator.bits_per_symbol
    system = (
        modulator(),
        *awgn_link(es_n0),
        DecisionDirected(
            modulator(),
            demodulator(),
            DDPR_BUFFER_SIZE,
            SYMBOL_RATE,
            LASER_LINEWIDTH_ESTIMATE,
            es_n0,
        ),
    )
    return simulate_impl(system, length)


def nonlinear_link(tx_power_dbm: float) -> Sequence[Component]:
    return (
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        IQModulator(NoisyLaser(0, SYMBOL_RATE * CHANNEL_SPS)),
        SetPower(tx_power_dbm),
        SSFChannel(SPLITTING_POINT, SYMBOL_RATE * CHANNEL_SPS),
        Splitter(CONSUMERS),
        SSFChannel(FIBRE_LENGTH - SPLITTING_POINT, SYMBOL_RATE * CHANNEL_SPS),
        NoisyOpticalFrontEnd(SYMBOL_RATE * CHANNEL_SPS),
        Decimate(CHANNEL_SPS // RECEIVER_SPS),
        CDCompensator(FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, CDC_TAPS),
        PulseFilter(RECEIVER_SPS, down=RECEIVER_SPS),
    )


def experiment_link(rx_power_dbm: float) -> Sequence[Component]:
    # TODO running with the noise-less HeterodyneFrontEnd, there seems to be a
    # noise floor just under 1e-3. Is that due to phase noise? Removing phase
    # noise drops it to 1e-4 I think.
    return (
        AlamoutiEncoder(),
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        DPModulator(NoisyLaser(0, SYMBOL_RATE * CHANNEL_SPS)),
        SetPower(11.5),
        SSFChannel(FIBRE_LENGTH, SYMBOL_RATE * CHANNEL_SPS),
        SetPower(rx_power_dbm),
        PolarizationRotation(),
        DropPolarization(),
        NoisyHeterodyneFrontEnd(26, SYMBOL_RATE * CHANNEL_SPS),
        Decimate(CHANNEL_SPS // (2 * RECEIVER_SPS)),
        Digital90degHybrid(
            26, SYMBOL_RATE * 2 * RECEIVER_SPS
        ),  # 4 SpS to avoid aliasing.
        Decimate(2),  # Down to 2 SpS now.
        # CDCompensator(FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, CDC_TAPS),
        PulseFilter(RECEIVER_SPS, down=1),
    )


def make_experiment_simulation(
    modulator: Type[Modulator],
    demodulator: Type[Demodulator],
    length: int,
    rx_power_dbm: float,
) -> float:
    data_stream = PseudoRandomStream(ignore_first=True)

    def get_training_symbols() -> NDArray[np.cdouble]:
        # Need to return up-to-date symbols on each iteration.
        assert data_stream.last_chunk is not None
        return modulator()(data_stream.last_chunk)

    system = (
        modulator(),
        *experiment_link(rx_power_dbm),
        NormalizePower(),  # Match incoming data with training data.
        AdaptiveEqualizerAlamouti(
            21,
            1e-3,
            0.2,  # Adjust paremeters.
            modulator(),
            demodulator(),
            get_training_symbols,
            False,
        ),
        demodulator(),
    )

    return build_system(data_stream, system)(length)


def make_nonlinear_simulation(
    modulator: Type[Modulator],
    demodulator: Type[Demodulator],
    length: int,
    tx_power_dbm: float,
) -> float:
    system = (
        modulator(),
        *nonlinear_link(tx_power_dbm),
        DecisionDirected(
            modulator(),
            demodulator(),
            DDPR_BUFFER_SIZE,
            SYMBOL_RATE,
            LASER_LINEWIDTH_ESTIMATE,
            SNR_ESTIMATE,
        ),
    )
    return simulate_impl(system, length)


def run_awgn_simulation(
    p_executor: Optional[ProcessPoolExecutor],
    ber_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    simulation: Callable[[int, float], float],
) -> tuple[NDArray[np.int64], list[float]]:
    MAX_LENGTH = 2**18
    MAX_EB_N0_DB = 12

    eb_n0_dbs = np.arange(1, MAX_EB_N0_DB + 1)
    eb_n0s = energy_db_to_lin(eb_n0_dbs)

    # Only simulate up to the target BER.
    expected_bers = filter(lambda ber: ber > TARGET_BER, ber_function(eb_n0s))

    # Magic heuristic that estimates how many samples we need to get a decent
    # BER estimate. Rounds the result to the next greatest power of 2, as we
    # will be taking the FFT of the data later.
    lengths = [min(next_power_of_2(int(1200 / i)), MAX_LENGTH) for i in expected_bers]

    # TODO would be nice if this returned the iterator and the plot updated as
    # the results came in.
    bers = list(
        p_executor.map(simulation, lengths, eb_n0s)
        if p_executor
        else map(simulation, lengths, eb_n0s)
    )
    return eb_n0_dbs[: len(bers)], bers


def run_nonlinear_simulation(
    p_executor: Optional[ProcessPoolExecutor],
    simulation: Callable[[int, float], float],
) -> tuple[NDArray[np.float64], list[float]]:
    LENGTH = 2**18

    tx_power_dbms = np.linspace(-10, 15, 8, endpoint=True)
    lengths = cycle((LENGTH,))

    # TODO would be nice if this returned the iterator and the plot updated as
    # the results came in.
    bers = list(
        p_executor.map(simulation, lengths, tx_power_dbms)
        if p_executor
        else map(simulation, lengths, tx_power_dbms)
    )
    return tx_power_dbms, bers


def run_experiment_simulation(
    p_executor: Optional[ProcessPoolExecutor],
    simulation: Callable[[int, float], float],
) -> tuple[NDArray[np.float64], list[float]]:
    LENGTH = PulseFilter.symbols_for_total_length(2**18)

    rx_power_dbms = np.arange(-25, -18, dtype=np.float64)
    lengths = cycle((LENGTH,))

    bers = list(
        p_executor.map(simulation, lengths, rx_power_dbms)
        if p_executor
        else map(simulation, lengths, rx_power_dbms)
    )
    return rx_power_dbms, bers


def plot_cd_compensation_fir() -> None:
    """Replicate Figure 6 (the real parts, the imaginary parts, and the absolute
    values of the impulse response of the Chromatic Dispersion FIR compensation
    filter) from the paper.

    Parameters:
    N = 31
    Length = 500 km
    Sampling frequency = 21.4 GHz
    c = 3e8 m/s (we use the actual value)
    D = 17 ps/nm/km
    λ = 1553 nm (we use 1550 nm)
    L = 2
    """
    cdc = CDCompensator(500_000, 21.4e9, 2, 31)

    fig, axs = plt.subplots(nrows=3)
    fig.suptitle("CD Compensation FIR filter")
    fig.supylabel("Magnitude")
    fig.supxlabel("$n$")
    fig.subplots_adjust(hspace=0.75)
    axs[0].set_title("Real part")
    axs[0].stem(range(-15, 16), np.real(cdc.D))
    axs[1].set_title("Imaginary part")
    axs[1].stem(range(-15, 16), np.imag(cdc.D))
    axs[2].set_title("Absolute value")
    axs[2].stem(range(-15, 16), np.abs(cdc.D))
    plt.show()


def plot_cd_compensation_ber() -> None:
    """Find the optimal number of taps for the CD compensation filter, at
    Eb/N0 = 8 db (where the BER of 16-QAM over an AWGN is near 10**-2)."""
    EB_N0_dB = 8
    EB_N0 = energy_db_to_lin(EB_N0_dB)

    def simulation(fibre_length: int, taps: int) -> float:
        system = (
            Modulator16QAM(),
            PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
            ChromaticDispersion(fibre_length, SYMBOL_RATE * CHANNEL_SPS),
            Decimate(CHANNEL_SPS // RECEIVER_SPS),
            AWGN(EB_N0 * Modulator16QAM.bits_per_symbol, RECEIVER_SPS),
            CDCompensator(fibre_length, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, taps),
            PulseFilter(RECEIVER_SPS, down=RECEIVER_SPS),
            Demodulator16QAM(),
        )
        return simulate_impl(system, 2**18)

    # We can't pickle local functions, so we can't use an executor
    # unfortunately.
    sim_taps = np.arange(3, 75, 2)
    sim_bers_25 = list(map(simulation, cycle((25_000,)), sim_taps))
    sim_bers_50 = list(map(simulation, cycle((50_000,)), sim_taps))
    sim_bers_100 = list(map(simulation, cycle((100_000,)), sim_taps))

    _, ax = plt.subplots()

    th_ber = calculate_awgn_ber_with_16qam(np.asarray(EB_N0))

    ax.plot(sim_taps, sim_bers_25, alpha=0.6, linewidth=2, label="25 km", marker="o")
    ax.plot(sim_taps, sim_bers_50, alpha=0.6, linewidth=2, label="50 km", marker="o")
    ax.plot(sim_taps, sim_bers_100, alpha=0.6, linewidth=2, label="100 km", marker="o")

    xlims = ax.get_xlim()
    ax.hlines(th_ber, *xlims, label="Theoretical limit", color="purple")
    ax.set_xlim(xlims)

    ax.set_ylim(6e-3)
    ax.set_yscale("log")
    ax.set_ylabel("BER")
    ax.set_xlabel("CD compensation filter taps")
    ax.set_title(f"At $E_b/N_0 = {EB_N0_dB}$ dB")
    ax.legend()

    plt.show()


def plot_cd_compensation_freq_response() -> None:
    # Use a sufficiently large number of taps to get a decent graph.
    cdc = CDCompensator(FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, 99)
    plot_filter_with_gd(cdc.D)
    plot_filter_with_gd(cdc.h)


def plot_awgn_simulations(concurrent: bool = True) -> None:
    fig, ax = plt.subplots()

    markers = cycle(("o", "x", "s", "*"))
    labels = ("Simulated BPSK", "Simulated QPSK", "Simulated 16-QAM")
    simulations = (
        partial(make_awgn_simulation, ModulatorBPSK, DemodulatorBPSK),
        partial(make_awgn_simulation, ModulatorQPSK, DemodulatorQPSK),
        partial(make_awgn_simulation, Modulator16QAM, Demodulator16QAM),
    )
    ber_estimators = (
        calculate_awgn_ber_with_bpsk,
        calculate_awgn_ber_with_bpsk,
        calculate_awgn_ber_with_16qam,
    )

    if concurrent:
        with (
            ProcessPoolExecutor(max_workers=PHYSICAL_CORES) as p_executor,
            ThreadPoolExecutor(max_workers=len(simulations)) as t_executor,
        ):
            fn = partial(run_awgn_simulation, p_executor)
            sim_results = t_executor.map(fn, ber_estimators, simulations)
    else:
        fn = partial(run_awgn_simulation, None)
        sim_results = map(fn, ber_estimators, simulations)

    for label, marker, (eb_n0_dbs, bers) in zip(labels, markers, sim_results):
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
    ax.set_title("50 GBd, 25 km")
    ax.legend()

    fig.tight_layout()

    plt.show()


def plot_nonlinear_simulations(concurrent: bool = True) -> None:
    _, ax = plt.subplots()

    markers = cycle(("o", "x", "s", "*"))
    labels = ("Simulated 16-QAM",)
    simulations = (
        partial(make_nonlinear_simulation, Modulator16QAM, Demodulator16QAM),
    )

    if concurrent:
        with (
            ProcessPoolExecutor(max_workers=PHYSICAL_CORES) as p_executor,
            ThreadPoolExecutor(max_workers=len(simulations)) as t_executor,
        ):
            fn = partial(run_nonlinear_simulation, p_executor)
            sim_results = t_executor.map(fn, simulations)
    else:
        fn = partial(run_nonlinear_simulation, None)
        sim_results = map(fn, simulations)

    for label, marker, (eb_n0_dbs, bers) in zip(labels, markers, sim_results):
        ax.plot(eb_n0_dbs, bers, alpha=0.6, label=label, marker=marker)

    ax.set_ylim(TARGET_BER / 4)
    ax.set_yscale("log")
    ax.set_ylabel("BER")
    ax.set_xlabel("Launch power [dBm]")
    ax.legend()

    plt.show()


def plot_experiment_simulations(concurrent: bool = True) -> None:
    _, ax = plt.subplots()

    markers = cycle(("o", "x", "s", "*"))
    labels = ("Simulated 16-QAM",)
    simulations = (
        partial(make_experiment_simulation, Modulator16QAM, Demodulator16QAM),
    )

    if concurrent:
        with (
            ProcessPoolExecutor(max_workers=PHYSICAL_CORES) as p_executor,
            ThreadPoolExecutor(max_workers=len(simulations)) as t_executor,
        ):
            fn = partial(run_experiment_simulation, p_executor)
            sim_results = t_executor.map(fn, simulations)
    else:
        fn = partial(run_experiment_simulation, None)
        sim_results = map(fn, simulations)

    for label, marker, (eb_n0_dbs, bers) in zip(labels, markers, sim_results):
        ax.plot(eb_n0_dbs, np.log10(bers), alpha=0.6, label=label, marker=marker)

    orig_lims = ax.get_xlim()
    ax.hlines(-2, *orig_lims, label="FEC limit", alpha=0.4, linewidth=4)
    ax.set_xlim(orig_lims)

    # BER should be a straight line on a semi-log plot.
    ax.set_yscale("symlog", linthresh=0.1)
    yticks = np.arange(-2.6, -1.6, 0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(map("{:.1f}".format, yticks))

    ax.set_ylabel("BER")
    ax.set_xlabel("Received power [dBm]")
    ax.set_title("16-QAM, 50 GBd, 25 km")
    ax.legend()

    plt.show()


def plot_rrc() -> None:
    SPS = 2
    EB_N0_dB = 10
    EB_N0 = energy_db_to_lin(EB_N0_dB)
    SPANS = np.arange(2, 132, 4)
    BETA = 0.01
    LENGTH = 2**21

    fig, axs = plt.subplots(nrows=2)

    # AWGN limit and simulated BER over a range of spans.
    th_ber_16qam = calculate_awgn_ber_with_16qam(EB_N0)
    axs[0].axhline(
        th_ber_16qam, color="red", alpha=0.2, linewidth=5, label="AWGN limit"
    )

    bers = []
    for span in SPANS:
        # Need a new instance every time, as the impulse response is cached.
        pf_up = PulseFilter(SPS, up=SPS)
        pf_up.SPAN = span
        pf_up.BETA = BETA
        pf_down = PulseFilter(SPS, down=SPS)
        pf_down.SPAN = span
        pf_down.BETA = BETA

        system = build_system(
            PseudoRandomStream(),
            (
                Modulator16QAM(),
                pf_up,
                AWGN(EB_N0 * Modulator16QAM.bits_per_symbol, SPS),
                pf_down,
                Demodulator16QAM(),
            ),
        )

        bers.append(system(LENGTH))

    axs[0].plot(SPANS, bers, alpha=0.6, label="Simulated", marker="o")

    axs[0].set_yscale("log")
    axs[0].set_ylabel("BER")
    axs[0].set_xlabel("RRC Span [in symbols]")
    axs[0].legend()

    # Plot the frequency spectrum at the given β.
    pf = PulseFilter(SPS, up=SPS)
    pf.SPAN = 128
    pf.BETA = BETA
    axs[1].magnitude_spectrum(
        pf.impulse_response.tolist(),
        SYMBOL_RATE * SPS / 1e9,
        sides="twosided",
        scale="dB",
    )

    axs[1].set_xlabel("Frequency [GHz]")
    axs[1].set_ylabel("Magnitude [dB]")

    fig.suptitle(
        f"$E_b/N_0 = {EB_N0_dB}$ dB, $\\beta = {BETA}$, "
        f"16-QAM at {SYMBOL_RATE//10**9} GBd"
    )
    fig.tight_layout()

    plt.show()


def plot_cd_demo() -> None:
    SPS = 512
    data = np.asarray([0, 2, 1, 2, 0])

    pf = PulseFilter(SPS, up=SPS)

    x = pf(data)
    CD = ChromaticDispersion(FIBRE_LENGTH, SYMBOL_RATE * SPS)
    y = CD(x)

    fig, axs = plt.subplots(nrows=2)

    axs[0].plot(np.abs(x))
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Magnitude")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Transmitted pulses")

    axs[1].plot(np.abs(y))
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("After Chromatic Dispersion")

    fig.tight_layout()

    plt.show()


def plot_cd_ber_comparison() -> None:
    LENGTH = 2**18
    MAX_EB_N0_DB = 10

    def uncompensated_link(fibre_length: int, es_n0: float) -> list[Component]:
        return [
            Modulator16QAM(),
            PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
            ChromaticDispersion(fibre_length, SYMBOL_RATE * CHANNEL_SPS),
            Decimate(CHANNEL_SPS // RECEIVER_SPS),
            AWGN(es_n0, RECEIVER_SPS),
            PulseFilter(RECEIVER_SPS, down=RECEIVER_SPS),
            Demodulator16QAM(),
        ]

    def compensated_link(fibre_length: int, es_n0: float) -> list[Component]:
        link = uncompensated_link(fibre_length, es_n0)
        link.insert(
            -2,  # After AWGN, before PulseFilter.
            CDCompensator(
                fibre_length, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, CDC_TAPS
            ),
        )
        return link

    def plot_one(ax: Axes, fibre_length_km: int) -> None:
        markers = cycle(("o", "x", "s", "*"))
        labels = ("Uncompensated", f"Compensated ({CDC_TAPS} taps)")

        eb_n0_dbs = np.arange(1, MAX_EB_N0_DB + 1)
        eb_n0s = energy_db_to_lin(eb_n0_dbs)

        for label, marker, link in zip(
            labels, markers, (uncompensated_link, compensated_link)
        ):
            bers = []

            for eb_n0 in eb_n0s:
                system = build_system(
                    PseudoRandomStream(),
                    link(
                        fibre_length_km * 1000, eb_n0 * Modulator16QAM.bits_per_symbol
                    ),
                )

                bers.append(system(LENGTH))

            ax.plot(eb_n0_dbs, bers, alpha=0.6, label=label, marker=marker)

        th_ber_16qam = calculate_awgn_ber_with_16qam(eb_n0s)
        ax.plot(eb_n0_dbs, th_ber_16qam, alpha=0.2, linewidth=5, label="Theoretical")

        ax.set_xlabel("$E_b/N_0$ [dB]")
        ax.legend()
        ax.set_title(f"{fibre_length_km} km")

    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(6.4 * 1.5, 4.8))

    plot_one(axs[0], 1)
    plot_one(axs[1], 25)

    # Shared y-axis.
    axs[0].set_yscale("log")
    axs[0].set_ylabel("BER")

    fig.suptitle(f"16-QAM at {SYMBOL_RATE//10**9} GBd")
    fig.tight_layout()

    plt.show()


def plot_dd_phase_recovery_buffer_size() -> None:
    LENGTH = 2**14
    EB_N0_dB = 10
    EB_N0 = energy_db_to_lin(EB_N0_dB)

    laser = NoisyLaser(10, SYMBOL_RATE)
    buffer_sizes = np.arange(2, 17, 2)

    mod = Modulator16QAM()
    demod = Demodulator16QAM()

    link_common = [
        mod,
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        IQModulator(laser),
        ChromaticDispersion(FIBRE_LENGTH, SYMBOL_RATE * CHANNEL_SPS),
        OpticalFrontEnd(),
        Decimate(CHANNEL_SPS // RECEIVER_SPS),
        AWGN(EB_N0 * Modulator16QAM.bits_per_symbol, RECEIVER_SPS),
        CDCompensator(FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, CDC_TAPS),
        PulseFilter(RECEIVER_SPS, down=RECEIVER_SPS),
    ]

    bers = []
    for buffer_size in buffer_sizes:
        link = link_common + [
            DecisionDirected(
                mod,
                demod,
                buffer_size,
                SYMBOL_RATE,
                LASER_LINEWIDTH_ESTIMATE,
                EB_N0_dB,
            )
        ]

        system = build_system(PseudoRandomStream(), link)
        bers.append(system(LENGTH))

    _, axs = plt.subplots(ncols=2)

    axs[0].plot(buffer_sizes, bers, alpha=0.6, label="Simulated", marker="o")

    # Also plot AWGN limit.
    th_ber_16qam = calculate_awgn_ber_with_16qam(EB_N0)
    axs[0].axhline(
        th_ber_16qam, color="red", alpha=0.2, linewidth=5, label="AWGN limit"
    )

    axs[0].set_yscale("log")
    axs[0].set_ylabel("BER")
    axs[0].set_xlabel("Phase recovery buffer size")
    axs[0].set_title(f"$E_b/N_0 = {EB_N0_dB}$ dB, 16-QAM at {SYMBOL_RATE//10**9} GBd")
    axs[0].legend()

    # Plot ML filter on the side.
    ml_filter = DecisionDirected(
        mod, demod, 16, SYMBOL_RATE, LASER_LINEWIDTH_ESTIMATE, EB_N0_dB
    ).ml_filter
    axs[1].stem(ml_filter)

    axs[1].set_xlabel("Taps")
    axs[1].set_ylabel("Filter coefficient")
    axs[1].set_title("ML filter (16 taps)")

    plt.show()


def plot_step_size_comparison() -> None:
    LENGTH = 2**16
    TX_POWER_dBm = 25

    hs = [50, 100, 250, 500, 1000, 2000, FIBRE_LENGTH]

    channel = SSFChannel(FIBRE_LENGTH, SYMBOL_RATE * CHANNEL_SPS)

    system = build_system(
        PseudoRandomStream(),
        (
            Modulator16QAM(),
            PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
            IQModulator(ContinuousWaveLaser(TX_POWER_dBm)),
            channel,
            NoisyOpticalFrontEnd(SYMBOL_RATE * CHANNEL_SPS),
            Decimate(CHANNEL_SPS // RECEIVER_SPS),
            CDCompensator(
                FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, CDC_TAPS
            ),
            PulseFilter(RECEIVER_SPS, down=RECEIVER_SPS),
            Demodulator16QAM(),
        ),
    )

    bers = []
    for h in hs:
        channel.h = h
        bers.append(system(LENGTH))

    fig, ax = plt.subplots()

    ax.plot(hs, bers, alpha=0.6, marker="o")

    ax.set_yscale("log")
    ax.set_ylabel("BER")
    ax.set_xscale("log")
    ax.set_xlabel("Split-step Fourier step size [m]")
    ax.set_title(f"TX power = {TX_POWER_dBm} dBm, 16-QAM at {SYMBOL_RATE//10**9} GBd")

    fig.tight_layout()

    plt.show()


def plot_adaptive_equalizer_comparison() -> None:
    def make_simulation(
        adaptive: bool,
        cdc_taps: int,
        length: int,
        tx_power_dbm: float,
    ) -> float:
        link = [
            Modulator16QAM(),
            PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
            IQModulator(NoisyLaser(tx_power_dbm, SYMBOL_RATE * CHANNEL_SPS)),
            SSFChannel(SPLITTING_POINT, SYMBOL_RATE * CHANNEL_SPS),
            Splitter(CONSUMERS),
            SSFChannel(FIBRE_LENGTH - SPLITTING_POINT, SYMBOL_RATE * CHANNEL_SPS),
            NoisyOpticalFrontEnd(SYMBOL_RATE * CHANNEL_SPS),
            Decimate(CHANNEL_SPS // RECEIVER_SPS),
            CDCompensator(
                FIBRE_LENGTH, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, cdc_taps
            ),
            # Adaptive Equalizer already downsamples 2 to 1.
            PulseFilter(RECEIVER_SPS, down=1 if adaptive else 2),
            DecisionDirected(
                Modulator16QAM(),
                Demodulator16QAM(),
                DDPR_BUFFER_SIZE,
                SYMBOL_RATE,
                LASER_LINEWIDTH_ESTIMATE,
                SNR_ESTIMATE,
            ),
        ]

        if adaptive:
            link.insert(-1, AdaptiveEqualizer(63, 1e-3))

        return simulate_impl(link, length)

    markers = cycle(("o", "x", "s", "*"))
    labels = (
        f"Adaptive: 63 taps, CDC: {CDC_TAPS // 2} taps",
        f"Adaptive: 63 taps, CDC: {CDC_TAPS} taps",
        f"Adaptive: off taps, CDC: {CDC_TAPS // 2} taps",
        f"Adaptive: off taps, CDC: {CDC_TAPS} taps",
    )
    simulations = (
        partial(make_simulation, True, CDC_TAPS // 2),
        partial(make_simulation, True, CDC_TAPS),
        partial(make_simulation, False, CDC_TAPS // 2),
        partial(make_simulation, False, CDC_TAPS),
    )

    fn = partial(run_nonlinear_simulation, None)
    sim_results = map(fn, simulations)

    _, ax = plt.subplots()

    for label, marker, (eb_n0_dbs, bers) in zip(labels, markers, sim_results):
        ax.plot(eb_n0_dbs, bers, alpha=0.6, label=label, marker=marker)

    ax.set_yscale("log")
    ax.set_ylabel("BER")
    # TODO use the power going into the fibre (after the I/Q Modulator).
    ax.set_xlabel("Power at modulator input (dBm)")
    ax.legend()

    plt.show()


def plot_phase_noise() -> None:
    class Plotter(Component):
        @property
        def input_type(self) -> tuple[Signal, Type, int | None]:
            return Signal.SYMBOLS, np.cdouble, None

        @property
        def output_type(self) -> tuple[Signal, Type, int | None]:
            return Signal.SYMBOLS, np.cdouble, None

        def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
            plt.scatter(np.real(symbols), np.imag(symbols), s=0.1)
            plt.xlabel("In-phase")
            plt.ylabel("Quadrature")
            plt.title("16-QAM constellation corrupted by phase noise")
            plt.show()

            return symbols

    link = (
        Modulator16QAM(),
        PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS),
        IQModulator(NoisyLaser(10, SYMBOL_RATE * CHANNEL_SPS)),
        OpticalFrontEnd(),
        Decimate(CHANNEL_SPS // RECEIVER_SPS),
        AWGN(energy_db_to_lin(40), RECEIVER_SPS),  # Very mild AWGN.
        PulseFilter(RECEIVER_SPS, down=RECEIVER_SPS),
        NormalizePower(),
        Plotter(),
        Demodulator16QAM(),
    )

    simulate_impl(link, 2**15)


def plot_phase_noise_sample() -> None:
    nl = NoisyLaser(0, 100e9)
    xs = np.arange(2048) / 100

    for _ in range(4):
        nl.sample_phase_noise(2048)
        plt.plot(xs, nl.last_noise)

    plt.title(r"Phase noise samples, $\Delta\nu = 100$ kHz")
    plt.ylabel("Phase noise [rad]")
    plt.xlabel("Time [ns]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_experiment_simulations()

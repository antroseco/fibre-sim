from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt

from channel import AWGN
from data_stream import PseudoRandomStream
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
    calculate_awgn_ber_with_bpsk,
    calculate_awgn_ser_with_qam,
    plot_ber,
)


def calculate_n0(eb_n0: float, bits_per_symbol: int) -> float:
    # Energy per symbol.
    es_n0 = eb_n0 * bits_per_symbol

    # Each symbol has unit energy, so N0 is just the reciprocal.
    return 1 / es_n0


def simulate_impl(system: Sequence[Component], length: int) -> float:
    return build_system(PseudoRandomStream(), system)(length) / length


def simulate_bpsk(length: int, eb_n0: float) -> float:
    # BPSK over AWGN channel.
    N0 = calculate_n0(eb_n0, 1)
    system = (ModulatorBPSK(), AWGN(N0), DemodulatorBPSK())
    return simulate_impl(system, length)


def simulate_qpsk(length: int, eb_n0: float) -> float:
    # QPSK over AWGN channel.
    N0 = calculate_n0(eb_n0, 2)
    system = (ModulatorQPSK(), AWGN(N0), DemodulatorQPSK())
    return simulate_impl(system, length)


def simulate_16qam(length: int, eb_n0: float) -> float:
    # 16-QAM over AWGN channel.
    N0 = calculate_n0(eb_n0, 4)
    system = (Modulator16QAM(), AWGN(N0), Demodulator16QAM())
    return simulate_impl(system, length)


if __name__ == "__main__":
    LENGTH = 10**6

    eb_n0_db = np.arange(1, 8, 0.5)
    eb_n0 = 10 ** (eb_n0_db / 10)

    th_ber_psk = calculate_awgn_ber_with_bpsk(eb_n0)
    # This is the SER. Divide by 4 (bits per symbol) to get the approximate BER.
    th_ber_16qam = calculate_awgn_ser_with_qam(16, eb_n0) / 4
    ber_bpsk = [simulate_bpsk(LENGTH, i) for i in eb_n0]
    ber_qpsk = [simulate_qpsk(LENGTH, i) for i in eb_n0]
    ber_16qam = [simulate_16qam(LENGTH, i) for i in eb_n0]

    _, ax = plt.subplots()
    plot_ber(
        ax,
        eb_n0_db,
        (th_ber_psk, ber_bpsk, ber_qpsk, th_ber_16qam, ber_16qam),
        (
            "Theoretical BPSK/QPSK",
            "Simulated BPSK",
            "Simulated QPSK",
            "Theoretical 16-QAM",
            "Simulated 16-QAM",
        ),
    )

    plt.show()

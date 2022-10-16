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
from utils import calculate_awgn_ber_with_bpsk, calculate_awgn_ser_with_qam, plot_ber


def simulate_bpsk(length: int, N0: float) -> int:
    # BPSK over AWGN channel.
    system = (ModulatorBPSK(), AWGN(N0), DemodulatorBPSK())
    return build_system(PseudoRandomStream(), system)(length)


def simulate_qpsk(length: int, N0: float) -> int:
    # QPSK over AWGN channel.
    system = (ModulatorQPSK(), AWGN(N0 / 2), DemodulatorQPSK())
    return build_system(PseudoRandomStream(), system)(length)


def simulate_16qam(length: int, N0: float) -> int:
    # 16-QAM over AWGN channel.
    system = (Modulator16QAM(), AWGN(N0 / 4), Demodulator16QAM())
    return build_system(PseudoRandomStream(), system)(length)


if __name__ == "__main__":
    LENGTH = 10**6

    eb_n0_db = np.arange(1, 8, 0.5)
    eb_n0 = 10 ** (eb_n0_db / 10)

    th_ber_psk = calculate_awgn_ber_with_bpsk(eb_n0)
    # This is the SER. Divide by 4 (bits per symbol) to get the approximate BER.
    th_ber_16qam = calculate_awgn_ser_with_qam(16, eb_n0) / 4
    ber_bpsk = [simulate_bpsk(LENGTH, 1 / i) / LENGTH for i in eb_n0]
    ber_qpsk = [simulate_qpsk(LENGTH, 1 / i) / LENGTH for i in eb_n0]
    ber_16qam = [simulate_16qam(LENGTH, 1 / i) / LENGTH for i in eb_n0]

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

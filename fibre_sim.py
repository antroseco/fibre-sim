import numpy as np
from matplotlib import pyplot as plt

from channel import AWGN
from data_stream import PseudoRandomStream
from modulation import DemodulatorBPSK, DemodulatorQPSK, ModulatorBPSK, ModulatorQPSK
from system import build_system
from utils import calculate_awgn_ber_with_bpsk, plot_ber


def simulate_bpsk(length: int, N0: float) -> int:
    # BPSK over AWGN channel.
    system = (ModulatorBPSK(), AWGN(N0), DemodulatorBPSK())
    return build_system(PseudoRandomStream(), system)(length)


def simulate_qpsk(length: int, N0: float) -> int:
    # QPSK over AWGN channel.
    system = (ModulatorQPSK(), AWGN(N0 / 2), DemodulatorQPSK())
    return build_system(PseudoRandomStream(), system)(length)


if __name__ == "__main__":
    LENGTH = 10**6

    eb_n0_db = np.arange(1, 8, 0.5)
    eb_n0 = 10 ** (eb_n0_db / 10)

    theoretical_bers = calculate_awgn_ber_with_bpsk(eb_n0)
    bpsk_ber = [simulate_bpsk(LENGTH, 1 / i) / LENGTH for i in eb_n0]
    qpsk_ber = [simulate_qpsk(LENGTH, 1 / i) / LENGTH for i in eb_n0]

    _, ax = plt.subplots()
    plot_ber(
        ax,
        eb_n0_db,
        (theoretical_bers, bpsk_ber, qpsk_ber),
        ("Theoretical", "Simulated BPSK", "Simulated QPSK"),
    )

    plt.show()

import numpy as np
from matplotlib import pyplot as plt

from channel import AWGN
from data_stream import PseudoRandomStream
from modulation import DemodulatorBPSK, ModulatorBPSK
from system import build_system
from utils import calculate_awgn_ber_with_bpsk, plot_ber


def simulate(length: int, N0: float) -> tuple[int, int]:
    # BPSK over AWGN channel.
    system = (ModulatorBPSK(), AWGN(N0), DemodulatorBPSK())

    return build_system(PseudoRandomStream(1), system)(length)


if __name__ == "__main__":
    LENGTH = 10**6

    eb_n0_db = np.arange(1, 8, 0.5)
    eb_n0 = 10 ** (eb_n0_db / 10)

    theoretical_bers = calculate_awgn_ber_with_bpsk(eb_n0)
    bers = [simulate(LENGTH, 1 / i)[0] / LENGTH for i in eb_n0]

    print(bers)

    _, ax = plt.subplots()
    plot_ber(ax, eb_n0_db, (theoretical_bers, bers), ("Theoretical", "Simulation"))
    plt.show()

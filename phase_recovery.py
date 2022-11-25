import numpy as np
from numpy.typing import NDArray

from modulation import Demodulator, Modulator
from utils import Component, normalize_energy


from matplotlib import pyplot as plt


class BlindPhaseSearch(Component):
    B = 128  # Number of test rotations.
    N = 16  # Number of past and future symbols used.
    P = np.pi / 2  # For all M-QAM modulation schemes.

    def __init__(self, modulator: Modulator, demodulator: Demodulator) -> None:
        super().__init__()

        self.modulator = modulator
        self.demodulator = demodulator

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        b = np.arange(-self.B // 2, self.B // 2)
        assert b.size == self.B

        # FIXME just use np.linspace().
        thetas = b * (self.P / self.B)

        normalized = normalize_energy(symbols)

        rotated = [normalized * np.exp(1j * theta) for theta in thetas]

        distances = []
        for signal in rotated:
            decisions = normalize_energy(self.modulator(self.demodulator(symbols)))
            difference = signal - decisions
            distances.append(np.conj(difference) * difference)

        estimates = []
        for symbol_idx in range(len(normalized)):
            first = min(0, symbol_idx - self.N)
            last = min(len(normalized), symbol_idx + self.N + 1)

            theta_costs = [np.sum(distance[first:last]) for distance in distances]
            estimates.append(thetas[np.argmin(theta_costs)])

        unwrapped = []
        last_phase = 0
        for estimate in estimates:
            n = np.floor(
                0.5
                + (2**self.modulator.bits_per_symbol / (2 * np.pi))
                * (last_phase - estimate)
            )
            unwrapped.append(
                estimate + n * (2 * np.pi / 2**self.modulator.bits_per_symbol)
            )

        plt.plot(estimates, label="estimates")
        plt.plot(unwrapped, label="unwrapped")
        plt.legend()
        plt.show()

        return symbols * np.exp(1j * np.asarray(unwrapped))

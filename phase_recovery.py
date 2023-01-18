from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from modulation import Demodulator, Modulator
from utils import (
    Component,
    energy_db_to_lin,
    normalize_energy,
    signal_power,
    has_one_polarization,
)


class PhaseRecovery(Component):
    def __init__(self, modulator: Modulator, demodulator: Demodulator) -> None:
        super().__init__()

        self.modulator = modulator
        self.demodulator = demodulator


class BlindPhaseSearch(PhaseRecovery):
    B = 128  # Number of test rotations.
    N = 16  # Number of past and future symbols used.
    P = np.pi / 2  # For all M-QAM modulation schemes.

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(symbols)

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
            # TODO see if np.unwrap() works just as well.
            n = np.floor(
                0.5
                + (2**self.modulator.bits_per_symbol / (2 * np.pi))
                * (last_phase - estimate)
            )
            unwrapped.append(
                estimate + n * (2 * np.pi / 2**self.modulator.bits_per_symbol)
            )

        return symbols * np.exp(1j * np.asarray(unwrapped))


class DecisionDirected(PhaseRecovery):
    def __init__(
        self,
        modulator: Modulator,
        demodulator: Demodulator,
        buffer_size: int,
        symbol_rate: float,
        linewidth: float,
        snr_dB: float,
    ) -> None:
        super().__init__(modulator, demodulator)

        # Make sure we aren't inverting huge matrices by accident.
        assert 0 < buffer_size < 65
        self.buffer_size = buffer_size

        assert symbol_rate > 0
        self.symbol_period = 1 / symbol_rate

        assert linewidth > 0
        self.linewidth = linewidth

        assert snr_dB > 0
        self.snr = energy_db_to_lin(snr_dB)

        # Aids testing and debugging.
        self.last_estimates: Optional[NDArray[np.float64]] = None

        assert self.modulator.bits_per_symbol == self.demodulator.bits_per_symbol
        self.bits_per_symbol = self.modulator.bits_per_symbol

    @cached_property
    def ml_filter(self) -> NDArray[np.float64]:
        N = self.buffer_size

        # Phase noise estimate.
        phase_noise_var = 2 * np.pi * self.linewidth * self.symbol_period

        # Additive noise estimate.
        additive_noise_var = 1 / (2 * self.snr)

        # Covariance matrix.
        K: NDArray[np.float64] = np.fromfunction(np.minimum, (N, N))
        I = np.eye(N)

        C = phase_noise_var * K + additive_noise_var * I

        # Significantly faster way of solving for w_ml.
        w_ml = np.linalg.solve(C.T, np.ones(N))
        w_ml /= np.max(w_ml)

        return w_ml

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.bool8]:
        assert has_one_polarization(symbols)

        estimates = np.empty(symbols.size)
        decisions = np.empty((symbols.size, self.bits_per_symbol), dtype=np.bool8)

        shift_register = np.ones(self.buffer_size, np.cdouble)

        scale = np.sqrt(signal_power(symbols))

        for i in range(symbols.size):
            # Compute phase estimate.
            estimates[i] = np.angle(self.ml_filter @ shift_register)

            # Recover the next symbol.
            symbol: np.cdouble = symbols[i]
            compensated: np.cdouble = symbol * np.exp(-1j * estimates[i])

            # Need to modulate the decided bits again to recover their symbol.
            decisions[i] = self.demodulator(np.atleast_1d(compensated), scale)
            decided: np.cdouble = self.modulator(decisions[i])[0]

            # Advance shift register and insert the latest term to the front.
            shift_register = np.roll(shift_register, 1)
            prediction_term = symbol * np.conj(decided)
            shift_register[0] = prediction_term / np.abs(prediction_term)

        self.last_estimates = estimates

        # TODO unwrap the phase

        # Flatten demodulated bits.
        return np.ravel(decisions)

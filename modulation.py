from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from laser import Laser
from utils import Component, signal_power

from matplotlib import pyplot as plt


class Modulator(Component):
    input_type = "bits"
    output_type = "cd symbols"

    bits_per_symbol: int = 0

    @abstractmethod
    def __call__(self, data: NDArray[np.bool8]) -> NDArray[np.cdouble]:
        assert data.ndim == 1
        assert data.dtype == np.bool8
        assert data.size % self.bits_per_symbol == 0


class Demodulator(Component):
    input_type = "cd symbols"
    output_type = "bits"

    bits_per_symbol: int = 0

    @abstractmethod
    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.bool8]:
        assert symbols.dtype == np.cdouble


class ModulatorBPSK(Modulator):
    bits_per_symbol = 1

    def __call__(self, data: NDArray[np.bool8]) -> NDArray[np.cdouble]:
        super().__call__(data)

        # Map bits to symbols (1 -> -1, 0 -> 1).
        return 1 - 2 * data.astype(np.cdouble)


class DemodulatorBPSK(Demodulator):
    bits_per_symbol = 1

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.bool8]:
        super().__call__(symbols)

        return symbols < 0


class ModulatorQPSK(Modulator):
    bits_per_symbol = 2

    def __call__(self, data: NDArray[np.bool8]) -> NDArray[np.cdouble]:
        super().__call__(data)

        # Constellation has 4 symbols. Adjacent symbols only vary by 1 bit.
        #
        #         Q
        #   10    |    00
        #         |
        #  ---------------I
        #         |
        #   11    |    01
        #         |
        I = 1 - 2 * data[0::2]
        Q = 1j - 2j * data[1::2]

        # Normalize symbol energy.
        return (I + Q) / np.sqrt(2)


class DemodulatorQPSK(Demodulator):
    bits_per_symbol = 2

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.bool8]:
        super().__call__(symbols)

        # In-phase component is the MSB.
        msb = np.real(symbols) < 0
        # Quadrature component is the LSB.
        lsb = np.imag(symbols) < 0

        data = np.empty(msb.size + lsb.size, dtype=np.bool8)
        data[0::2] = msb
        data[1::2] = lsb

        return data


class Modulator16QAM(Modulator):
    bits_per_symbol = 4

    @staticmethod
    def impl(msbs: NDArray[np.bool8], lsbs: NDArray[np.bool8]) -> NDArray[np.float64]:
        assert msbs.size == lsbs.size

        # Looking at the two LSBs of the constellation symbols, we can see that
        # they completly determine the in-phase component. Looking at the two
        # MSBs, we can see that they determine the quadrature component. As they
        # follow they same Gray code each time (00 -> 01 -> 11 -> 10), we only
        # need to look at two bits at a time to determine each component.
        #
        # 3 boolean expressions are needed to determine each component, where
        # the msbs and lsbs below refer to the MSB and LSB in each pair of bits.
        # The first expression covers [01, 11, 10], the second [11, 10], and the
        # third [10] only. Hence, each bit pattern may match multiple
        # expressions. This should be considerably faster than matching one
        # expression for each pattern and then multiplying by e.g. 3.
        offsets = np.zeros_like(msbs, dtype=np.float64)
        offsets += msbs | lsbs
        offsets += msbs
        offsets += msbs & ~lsbs

        return offsets

    def __call__(self, data: NDArray[np.bool8]) -> NDArray[np.cdouble]:
        super().__call__(data)

        # Constellation has 16 symbols. Adjacent symbols only vary by 1 bit.
        # 0111 is at (1, 1) and 0010 is at (3, 3).
        #
        #              Q
        #   0000  0001 | 0011  0010
        #              |
        #              |
        #   0100  0101 | 0111  0110
        #              |
        #   ------------------------I
        #              |
        #   1100  1101 | 1111  1110
        #              |
        #              |
        #   1000  1001 | 1011  1010
        #              |
        #
        # In-phase component carries the 2 LSBs.
        I = -3 + 2 * self.impl(data[2::4], data[3::4])
        # Quadrature component carries the 2 MSBs.
        Q = 3j - 2j * self.impl(data[0::4], data[1::4])

        # Normalize symbol energy. The mean energy of the constellation is
        # defined as the expected value of |a+bj|^2. With a uniform
        # distribution over all symbols, this comes out to 10. Thus, we need to
        # divide the *amplitude* by the square root of 10.
        return (I + Q) / np.sqrt(10)


class Demodulator16QAM(Demodulator):
    bits_per_symbol = 4

    @staticmethod
    def impl(
        symbols: NDArray[np.float64], scale: float
    ) -> tuple[NDArray[np.bool8], NDArray[np.bool8]]:
        # FIXME explanation. Replace magic numbers.
        msbs = symbols > 0
        lsbs = np.abs(symbols) <= (2 * scale / np.sqrt(10))

        return msbs, lsbs

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.bool8]:
        super().__call__(symbols)

        # FIXME this should be replaced with a proper filter. Due to pulse
        # shaping and downsampling, the symbols no longer have unit energy. As
        # we rely on a threshold to distinguish between the inner and outer
        # constellation squares, we need to scale it based on the mean energy
        # of the received symbols.
        # FIXME replace with a function from utils.
        scale = np.sqrt(np.mean(np.abs(symbols) ** 2))

        # Each symbols carries 4 bits. The in-phase component contains the 2
        # LSBs, and the quadrature component contains the 2 MSBs.
        data = np.empty(symbols.size * self.bits_per_symbol, dtype=np.bool8)
        data[0::4], data[1::4] = self.impl(-np.imag(symbols), scale)
        data[2::4], data[3::4] = self.impl(np.real(symbols), scale)

        return data


class DemodulatorPR16QAM(Demodulator16QAM):
    def filt(self, symbols, N: int) -> NDArray[np.float64]:
        Es = signal_power(symbols)

        Ts = 1e-11  # TODO
        phase_noise_var = 2 * np.pi * 100e3 * Ts
        additive_noise_var = Es / (2 * 40)  # FIXME

        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = min(i, j)

        I = np.eye(N)
        C = Es**2 * phase_noise_var * K + Es * additive_noise_var * I

        w_ml = (np.ones(N).T @ np.linalg.inv(C)).T
        w_ml /= np.max(w_ml)
        return w_ml

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.bool8]:
        theta = np.zeros(symbols.size)
        decisions = np.empty(symbols.size * self.bits_per_symbol, dtype=np.bool8)

        N = 128  # Number of past symbols used.
        b = np.ones(N, np.cdouble)  # Buffer.

        ML_filter = self.filt(symbols, N)

        scale = np.sqrt(np.mean(np.abs(symbols) ** 2))

        def demod(symbol: np.cdouble):
            data = np.empty(self.bits_per_symbol, dtype=np.bool8)
            data[0], data[1] = self.impl(-np.imag(symbol), scale)
            data[2], data[3] = self.impl(np.real(symbol), scale)
            return data

        for i in range(symbols.size):
            theta[i] = np.angle(ML_filter.T @ b)

            v = symbols[i] * np.exp(-1j * theta[i])

            # FIXME ughh...
            decisions[4 * i : 4 * i + 4] = demod(v)
            decision = Modulator16QAM()(decisions[4 * i : 4 * i + 4])
            b = np.roll(b, 1)
            b[0:1] = (
                # FIXME clean this up too.
                symbols[i]
                * np.conj(decision)
                / np.abs(symbols[i] * np.conj(decision))
            )

        # plt.plot(theta, label="estimated")
        # plt.legend()
        # plt.show()

        return decisions


class IQModulator(Component):
    input_type = "cd symbols"
    output_type = "cd electric field"

    def __init__(self, laser: Laser) -> None:
        super().__init__()

        self.laser = laser

    def __call__(self, voltages: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert voltages.ndim == 1
        assert voltages.size > 0

        # Input voltage should range from -Vπ to +Vπ. Remove this restriction
        # from the caller by inferring Vπ.
        Vpi = max(np.max(np.abs(np.real(voltages))), np.max(np.abs(np.imag(voltages))))

        # FIXME increase Vpi to make the cosine function looks more linear. Is
        # there a better way to do this?
        Vpi *= 1.35

        # Add the DC bias of -Vπ, and divide by Vπ, which is 1.
        phi = (voltages - Vpi) * (np.pi / Vpi)

        Efield = 0.5 * self.laser(voltages.size)
        Efield *= np.cos(np.real(phi) / 2) + 1j * np.sin(np.imag(phi) / 2)

        return Efield

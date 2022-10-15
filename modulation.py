from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from utils import Component


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
        assert symbols.ndim == 1
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
        assert msbs.ndim == lsbs.ndim == 1

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

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

        # TODO explanation
        offsets = np.zeros_like(msbs, dtype=np.float64)
        offsets += msbs | lsbs
        offsets += msbs
        offsets += msbs & ~lsbs

        return offsets

    def __call__(self, data: NDArray[np.bool8]) -> NDArray[np.cdouble]:
        super().__call__(data)

        # TODO Constellation diagram.
        I = -3 + 2 * self.impl(data[2::4], data[3::4])
        Q = 3j - 2j * self.impl(data[0::4], data[1::4])

        # Normalize symbol energy. TODO explanation.
        return (I + Q) / np.sqrt(10)

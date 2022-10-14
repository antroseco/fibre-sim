from abc import abstractmethod

import numpy as np

from utils import Component


class Modulator(Component):
    input_type = "u8 data"
    output_type = "cd symbols"

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 1


class Demodulator(Component):
    input_type = "cd symbols"
    output_type = "u8 data"

    @abstractmethod
    def __call__(self, symbols: np.ndarray) -> np.ndarray:
        assert symbols.ndim == 1


class ModulatorBPSK(Modulator):
    def __call__(self, data: np.ndarray) -> np.ndarray:
        super().__call__(data)

        assert data.min() >= 0
        assert data.max() <= 1

        # Map bits to symbols (1 -> -1, 0 -> 1).
        return 1 - 2 * data.astype(np.cdouble)


class DemodulatorBPSK(Demodulator):
    def __call__(self, symbols: np.ndarray) -> np.ndarray:
        super().__call__(symbols)

        return (symbols < 0).astype(np.uint8)


class ModulatorQPSK(Modulator):
    def __call__(self, data: np.ndarray) -> np.ndarray:
        super().__call__(data)

        assert data.min() >= 0
        assert data.max() <= 3

        # Constellation has 4 symbols. Adjacent symbols only vary by 1 bit.
        #         Q
        #   01    |    00
        #         |
        #  ---------------I
        #         |
        #   11    |    10
        #         |
        I = 1 - 2 * (data & 0b01).astype(np.bool8)
        Q = 1j - 2j * (data & 0b10).astype(np.bool8)

        print(I)
        print(Q)

        # Normalize symbol energy.
        return (I + Q) / np.sqrt(2)


class DemodulatorQPSK(Demodulator):
    def __call__(self, symbols: np.ndarray) -> np.ndarray:
        super().__call__(symbols)

        # In-phase component is the LSB.
        data = (symbols.real < 0).astype(np.uint8)
        # Quadrature component is the MSB.
        data |= (symbols.imag < 0).astype(np.uint8) << 1

        return data

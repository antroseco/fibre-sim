from abc import abstractmethod

import numpy as np

from utils import Component


class Modulator(Component):
    input_type = "u8 data"
    output_type = "cd symbols"

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 1


class ModulatorBPSK(Modulator):
    def __call__(self, data: np.ndarray) -> np.ndarray:
        super().__call__(data)

        assert data.min() >= 0
        assert data.max() <= 1

        # Map bits to symbols (1 -> -1, 0 -> 1).
        return 1 - 2 * data.astype(np.cdouble)


class Demodulator(Component):
    input_type = "cd symbols"
    output_type = "u8 data"

    @abstractmethod
    def __call__(self, symbols: np.ndarray) -> np.ndarray:
        assert symbols.ndim == 1


class DemodulatorBPSK(Demodulator):
    def __call__(self, symbols: np.ndarray) -> np.ndarray:
        super().__call__(symbols)

        return (symbols < 0).astype(np.uint8)

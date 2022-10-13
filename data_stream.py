from abc import ABC, abstractmethod

import numpy as np


class DataStream(ABC):
    input_type = "u8 data"
    output_type = "u8 data"

    def __init__(self) -> None:
        super().__init__()

        self.bit_errors = 0
        self.symbol_errors = 0

    @abstractmethod
    def generate(self, length: int) -> np.ndarray:
        pass

    @abstractmethod
    def validate(self, data: np.ndarray) -> int:
        pass


class PseudoRandomStream(DataStream):
    def __init__(self, bits_per_symbol: int) -> None:
        super().__init__()

        assert bits_per_symbol > 0
        assert 2**bits_per_symbol <= np.iinfo(np.uint8).max + 1

        self.bits_per_symbol = bits_per_symbol

        self.last_chunk = None
        self.rng = np.random.default_rng()

    def generate(self, length: int) -> np.ndarray:
        self.last_chunk = self.rng.integers(
            0, 2**self.bits_per_symbol, length, dtype=np.uint8
        )

        return self.last_chunk

    def validate(self, data: np.ndarray) -> None:
        assert self.last_chunk is not None
        assert data.size == self.last_chunk.size

        self.bit_errors += np.count_nonzero(
            np.unpackbits(data) ^ np.unpackbits(self.last_chunk)
        )
        self.symbol_errors += np.count_nonzero(data ^ self.last_chunk)

from abc import abstractmethod
from typing import Optional, Type

import numpy as np
from numpy.typing import NDArray

from utils import Signal, TypeChecked


class DataStream(TypeChecked):
    def __init__(self) -> None:
        super().__init__()

        self.bit_errors: int = 0
        self.bits_validated: int = 0

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.BITS, np.bool_, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.BITS, np.bool_, None

    @abstractmethod
    def generate(self, length: int) -> NDArray[np.bool_]:
        pass

    @abstractmethod
    def validate(self, data: NDArray[np.bool_]) -> int:
        pass

    def reset(self) -> None:
        self.bit_errors = 0

    @property
    def ber(self) -> float:
        return self.bit_errors / self.bits_validated


class PseudoRandomStream(DataStream):
    def __init__(self, ignore_first: bool = False) -> None:
        super().__init__()

        self.ignore_next = ignore_first

        self.last_chunk: Optional[NDArray[np.bool_]] = None
        self.rng = np.random.default_rng()

    def generate(self, length: int) -> NDArray[np.bool_]:
        self.last_chunk = self.rng.integers(
            0, 1, endpoint=True, size=length, dtype=np.bool_
        )

        return self.last_chunk

    def validate(self, data: NDArray[np.bool_]) -> None:
        assert self.last_chunk is not None
        assert data.size == self.last_chunk.size

        if self.ignore_next:
            self.ignore_next = False
            return

        self.bit_errors += np.count_nonzero(data ^ self.last_chunk)
        self.bits_validated += data.size

        # Prevent the last chunk from being reused accidentally.
        self.last_chunk = None

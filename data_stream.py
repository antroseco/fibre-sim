from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class DataStream(ABC):
    input_type = "bits"
    output_type = "bits"

    def __init__(self) -> None:
        super().__init__()

        self.bit_errors: int = 0

    @abstractmethod
    def generate(self, length: int) -> NDArray[np.bool_]:
        pass

    @abstractmethod
    def validate(self, data: NDArray[np.bool_]) -> int:
        pass

    def reset(self) -> None:
        self.bit_errors = 0


class PseudoRandomStream(DataStream):
    def __init__(self) -> None:
        super().__init__()

        self.last_chunk: Optional[NDArray[np.bool_]] = None
        self.rng = np.random.default_rng()
        self.lag_cache: dict[int, int] = {}

    def generate(self, length: int) -> NDArray[np.bool_]:
        self.last_chunk = self.rng.integers(
            0, 1, endpoint=True, size=length, dtype=np.bool_
        )

        return self.last_chunk

    def validate(self, data: NDArray[np.bool_]) -> None:
        assert self.last_chunk is not None
        assert data.size == self.last_chunk.size

        self.bit_errors += np.count_nonzero(data ^ self.last_chunk)

        # Prevent the last chunk from being reused accidentally.
        self.last_chunk = None

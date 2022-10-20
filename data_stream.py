from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, correlation_lags


class DataStream(ABC):
    input_type = "bits"
    output_type = "bits"

    def __init__(self) -> None:
        super().__init__()

        self.bit_errors: int = 0

    @abstractmethod
    def generate(self, length: int) -> NDArray[np.bool8]:
        pass

    @abstractmethod
    def validate(self, data: NDArray[np.bool8]) -> int:
        pass


class PseudoRandomStream(DataStream):
    def __init__(self) -> None:
        super().__init__()

        self.last_chunk: Optional[NDArray[np.bool8]] = None
        self.rng = np.random.default_rng()
        self.lag: Optional[int] = None

    def estimate_lag(self, data: NDArray[np.bool8]) -> int:
        assert self.last_chunk is not None

        corrs = correlate(
            data.astype(float), self.last_chunk.astype(float), method="fft"
        )
        lags = correlation_lags(data.size, self.last_chunk.size)

        # Return the lag that corresponds to the greatest cross-correlation
        # between the two signals.
        return lags[np.argmax(corrs)]

    def generate(self, length: int) -> NDArray[np.bool8]:
        self.last_chunk = self.rng.integers(
            0, 1, endpoint=True, size=length, dtype=np.bool8
        )

        return self.last_chunk

    def validate(self, data: NDArray[np.bool8]) -> None:
        assert self.last_chunk is not None
        assert data.size == self.last_chunk.size

        # Estimate this once and then cache it for future chunks. The lag should
        # not change. FIXME what if the length of the next chunk is different?
        if self.lag is None:
            self.lag = self.estimate_lag(data)

        self.bit_errors += np.count_nonzero(data ^ np.roll(self.last_chunk, self.lag))

        # Prevent the last chunk from being reused accidentally.
        self.last_chunk = None

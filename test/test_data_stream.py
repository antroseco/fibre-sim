import numpy as np
import pytest
from data_stream import PseudoRandomStream


def calculate_entropy(data: np.ndarray) -> float:
    _, counts = np.unique(data, return_counts=True)
    frequencies = counts / np.sum(counts)

    return -np.sum(frequencies * np.log2(frequencies))


class TestPseudoRandomStream:
    stream = PseudoRandomStream()

    def test_generate(self):
        LENGTH = 4096

        first = self.stream.generate(LENGTH)
        second = self.stream.generate(LENGTH)

        # Check dtype.
        assert first.dtype == second.dtype == np.bool8

        # Check that it respects the length argument.
        assert first.size == second.size == LENGTH

        # Different blocks should virtually never be the same.
        assert np.any(first != second)

        # Verify entropy.
        assert calculate_entropy(first) > 0.99
        assert calculate_entropy(second) > 0.99

    def test_validate(self):
        LENGTH = 8

        # validate() should throw if it's called before generate().
        with pytest.raises(Exception):
            self.stream.validate(np.zeros(LENGTH, dtype=np.bool8))

        # Now it should be fine; and it should report zero errors.
        data = self.stream.generate(LENGTH)
        self.stream.validate(data)

        assert self.stream.bit_errors == 0

        # It should throw if validate() is called consecutively.
        with pytest.raises(Exception):
            self.stream.validate(data)

        # It should also throw if the lengths don't match.
        self.stream.generate(LENGTH)
        with pytest.raises(Exception):
            self.stream.validate(np.zeros(2 * LENGTH, dtype=np.bool8))

        # Finally test if errors are counted correctly.
        self.stream.last_chunk = np.zeros(4, dtype=np.bool8)
        self.stream.validate(np.asarray((0, 0, 1, 1), dtype=np.bool8))

        assert self.stream.bit_errors == 2

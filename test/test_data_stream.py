import numpy as np
import pytest
from data_stream import PseudoRandomStream


def calculate_entropy(data: np.ndarray) -> float:
    _, counts = np.unique(data, return_counts=True)
    frequencies = counts / np.sum(counts)

    return -np.sum(frequencies * np.log2(frequencies))


class TestPseudoRandomStream:
    @staticmethod
    def test_generate():
        LENGTH = 4096
        stream = PseudoRandomStream()

        first = stream.generate(LENGTH)
        second = stream.generate(LENGTH)

        # Check dtype.
        assert first.dtype == second.dtype == np.bool8

        # Check that it respects the length argument.
        assert first.size == second.size == LENGTH

        # Different blocks should virtually never be the same.
        assert np.any(first != second)

        # Verify entropy.
        assert calculate_entropy(first) > 0.99
        assert calculate_entropy(second) > 0.99

    @staticmethod
    def test_validate():
        LENGTH = 8
        stream = PseudoRandomStream()

        # validate() should throw if it's called before generate().
        with pytest.raises(Exception):
            stream.validate(np.zeros(LENGTH, dtype=np.bool8))

        # Now it should be fine; and it should report zero errors.
        data = stream.generate(LENGTH)
        stream.validate(data)

        assert stream.bit_errors == 0

        # It should throw if validate() is called consecutively.
        with pytest.raises(Exception):
            stream.validate(data)

        # It should also throw if the lengths don't match.
        stream.generate(LENGTH)
        with pytest.raises(Exception):
            stream.validate(np.zeros(2 * LENGTH, dtype=np.bool8))

        # Finally test if errors are counted correctly.
        stream.last_chunk = np.zeros(4, dtype=np.bool8)
        stream.validate(np.asarray((0, 0, 1, 1), dtype=np.bool8))

        assert stream.bit_errors == 2

    def test_different_lengths(self):
        stream = PseudoRandomStream()

        # Test the *same* stream with data of different lengths. This checks
        # that the estimated lag is not cached incorrectly.
        for length in range(1, 1002, 100):
            data = stream.generate(length)
            stream.validate(data)

            assert stream.bit_errors == 0

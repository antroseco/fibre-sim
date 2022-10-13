import numpy as np
import pytest
from data_stream import PseudoRandomStream


def calculate_entropy(data: np.ndarray) -> float:
    _, counts = np.unique(data, return_counts=True)
    frequencies = counts / np.sum(counts)

    return -np.sum(frequencies * np.log2(frequencies))


class TestPseudoRandomStream:
    @staticmethod
    def test_init_bounds_checking():
        # Negative bits per symbol don't make sense.
        with pytest.raises(Exception):
            PseudoRandomStream(-1)
        # Neither do 0.
        with pytest.raises(Exception):
            PseudoRandomStream(0)
        # We use a uint8, so we are limited to 8 bits.
        with pytest.raises(Exception):
            PseudoRandomStream(9)

        # Anything between 1 and 8 bits per symbol should be fine.
        PseudoRandomStream(1)
        PseudoRandomStream(8)

    @staticmethod
    @pytest.mark.parametrize("bits", [1, 4, 8])
    def test_generate(bits):
        LENGTH = 4096
        stream = PseudoRandomStream(bits)

        first = stream.generate(LENGTH)
        second = stream.generate(LENGTH)

        # Check that it respects the length argument.
        assert first.size == second.size == LENGTH

        # Different blocks should virtually never be the same.
        assert np.any(first != second)

        # Check minimum and maximum values generated.
        assert first.min() >= 0
        assert first.max() <= 2**bits - 1

        # Verify entropy.
        assert np.isclose(calculate_entropy(first), bits, atol=0.1)

    @staticmethod
    def test_validate():
        LENGTH = 8
        stream = PseudoRandomStream(2)

        # validate() should throw if it's called before generate().
        with pytest.raises(Exception):
            stream.validate(np.zeros(LENGTH))

        # Now it should be fine; and it should report zero errors.
        data = stream.generate(LENGTH)
        stream.validate(data)

        assert stream.bit_errors == 0
        assert stream.symbol_errors == 0

        # It should throw if validate() is called consecutively.
        with pytest.raises(Exception):
            stream.validate(data)

        # It should also throw if the lengths don't match.
        stream.generate(LENGTH)
        with pytest.raises(Exception):
            stream.validate(np.zeros(2 * LENGTH))

        # Finally test if errors are counted correctly.
        stream.last_chunk = np.zeros(4, dtype=np.uint8)
        stream.validate(np.asarray((0, 0, 1, 3), dtype=np.uint8))

        assert stream.bit_errors == 3
        assert stream.symbol_errors == 2

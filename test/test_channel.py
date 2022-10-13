import numpy as np
import pytest
from channel import AWGN


class TestAWGN:
    test_data = np.arange(1000)

    def test_preserves_length(self):
        awgn = AWGN(1)

        assert awgn(self.test_data).size == self.test_data.size

    def test_symbols_modified(self):
        awgn = AWGN(1)

        noisy_data = awgn(self.test_data)

        assert np.all(noisy_data != self.test_data)
        assert np.isfinite(noisy_data).all()

    @pytest.mark.parametrize("N0", [1, 2, 3])
    def test_distribution(self, N0):
        awgn = AWGN(N0)

        noise = awgn(self.test_data) - self.test_data

        # Noise must have mean 0 and variance N0 / 2.
        assert np.isclose(noise.mean(), 0, atol=0.2)
        assert np.isclose(noise.var(), N0 / 2, atol=0.2)

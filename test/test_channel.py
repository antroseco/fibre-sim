import numpy as np
import pytest
from channel import AWGN


class TestAWGN:
    test_data = np.arange(2**12) + 2j

    def test_preserves_length(self):
        awgn = AWGN(1)

        assert awgn(self.test_data).size == self.test_data.size

    def test_symbols_modified(self):
        awgn = AWGN(1)

        noisy_data = awgn(self.test_data)

        assert np.all(np.real(noisy_data) != np.real(self.test_data))
        assert np.all(np.imag(noisy_data) != np.imag(self.test_data))
        assert np.isfinite(noisy_data).all()

    @pytest.mark.parametrize("N0", [1, 2, 3])
    def test_distribution(self, N0):
        awgn = AWGN(N0)

        noise = awgn(self.test_data) - self.test_data
        noise_r = np.real(noise)
        noise_i = np.imag(noise)

        # Noise must have mean 0 and variance N0 / 2.
        assert np.isclose(noise_r.mean(), 0, atol=0.2)
        assert np.isclose(noise_r.var(), N0 / 2, atol=0.2)
        assert np.isclose(noise_i.mean(), 0, atol=0.2)
        assert np.isclose(noise_i.var(), N0 / 2, atol=0.2)

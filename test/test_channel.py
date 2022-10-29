import numpy as np
import pytest
from channel import AWGN


class TestAWGN:
    rng = np.random.default_rng()
    test_data = rng.integers(-2, 2, endpoint=True, size=2**14) + 2j

    def test_preserves_length(self):
        awgn = AWGN(1, 1)

        assert awgn(self.test_data).size == self.test_data.size

    def test_symbols_modified(self):
        awgn = AWGN(1, 1)

        noisy_data = awgn(self.test_data)

        assert np.all(np.real(noisy_data) != np.real(self.test_data))
        assert np.all(np.imag(noisy_data) != np.imag(self.test_data))
        assert np.isfinite(noisy_data).all()

    @pytest.mark.parametrize("es_n0", (1, 2, 3))
    def test_distribution(self, es_n0):
        awgn = AWGN(es_n0, 1)

        noise = awgn(self.test_data) - self.test_data

        # Noise must have mean 0.
        assert np.isclose(noise.mean(), 0, atol=0.2)

        # Check Es/N0. N0 is the noise power spectral density, but white noise
        # has a flat power spectrum, equal to the variance of the Gaussian
        # distribution generating the noise.
        es = np.sum(np.abs(self.test_data) ** 2) / noise.size
        n0 = np.var(noise)
        assert np.isclose(es / n0, es_n0, atol=0.1)

    @staticmethod
    @pytest.mark.parametrize("samples_per_symbol", (1, 2, 3, 4))
    def test_samples_per_symbol(samples_per_symbol: int):
        ES = 2
        ES_N0 = 3
        SYMBOL_COUNT = 2**12

        # Construct a train of delta functions (so each symbol has energy 2).
        symbols = np.zeros(SYMBOL_COUNT * samples_per_symbol)
        symbols[::samples_per_symbol] = np.sqrt(ES)

        assert np.isclose(np.sum(np.abs(symbols) ** 2) / SYMBOL_COUNT, ES)

        awgn = AWGN(ES_N0, samples_per_symbol)
        noise = awgn(symbols) - symbols

        # Noise must have mean 0.
        assert np.isclose(noise.mean(), 0, atol=0.2)

        # Check Es/N0. N0 is the noise power spectral density, but white noise
        # has a flat power spectrum, equal to the variance of the Gaussian
        # distribution generating the noise.
        n0 = np.var(noise)
        assert np.isclose(ES / n0, ES_N0, atol=0.1)

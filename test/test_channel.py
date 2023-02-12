import numpy as np
import pytest

from channel import AWGN, PolarizationRotation, Splitter
from utils import energy_db_to_lin, signal_energy, signal_power


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
        es = signal_power(self.test_data)
        n0 = np.var(noise)
        assert np.isclose(es / n0, es_n0, atol=0.1)

    @staticmethod
    @pytest.mark.parametrize("samples_per_symbol", (1, 2, 3, 4))
    def test_samples_per_symbol(samples_per_symbol: int):
        ES = 2
        ES_N0 = 3
        SYMBOL_COUNT = 2**13

        # Construct a train of delta functions (so each symbol has energy 2).
        symbols = np.zeros(SYMBOL_COUNT * samples_per_symbol, dtype=np.cdouble)
        symbols[::samples_per_symbol] = np.sqrt(ES)

        assert np.isclose(signal_energy(symbols) / SYMBOL_COUNT, ES)

        awgn = AWGN(ES_N0, samples_per_symbol)
        noise = awgn(symbols) - symbols

        # Noise must have mean 0.
        assert np.isclose(noise.mean(), 0, atol=0.2)

        # Check Es/N0. N0 is the noise power spectral density, but white noise
        # has a flat power spectrum, equal to the variance of the Gaussian
        # distribution generating the noise.
        n0 = np.var(noise)
        assert np.isclose(ES / n0, ES_N0, atol=0.1)


class TestSplitter:
    rng = np.random.default_rng()
    test_data = rng.integers(-2, 2, endpoint=True, size=2**14) + 2j

    @pytest.mark.parametrize("ratio", [-1, 0, 1, 3, 17])
    def test_illegal_ratios(self, ratio: int) -> None:
        with pytest.raises(Exception):
            Splitter(ratio)

    def test_preserves_length(self) -> None:
        splitter = Splitter(32)

        assert splitter(self.test_data).size == self.test_data.size

    @pytest.mark.parametrize("ratio", [2, 4, 16, 32])
    def test_power_change(self, ratio: int) -> None:
        splitter = Splitter(ratio)

        power_in = signal_power(self.test_data)
        power_out = signal_power(splitter(self.test_data))

        # 3.5 dB loss per coupler (overhead of 0.5 dB).
        expected = 3.5 * np.log2(ratio)

        assert np.isclose(power_in / power_out, energy_db_to_lin(expected))


class TestPolarizationRotation:
    @staticmethod
    def test_conservation_of_energy():
        rot = PolarizationRotation()

        test_data = np.random.randn(128).reshape(2, 64).astype(np.cdouble)
        test_data += 2j * test_data

        rotated = rot(test_data)

        assert rotated.shape == test_data.shape
        assert rotated.dtype == test_data.dtype

        assert np.isclose(signal_energy(rotated), signal_energy(test_data))

    @staticmethod
    def test_common_rotation():
        # 90 degree rotation.
        rot = PolarizationRotation(np.pi / 2)

        test_data = np.arange(6, dtype=np.cdouble).reshape(2, 3)

        rotated = rot(test_data)

        assert rotated.shape == test_data.shape
        assert rotated.dtype == test_data.dtype

        assert np.isclose(signal_energy(rotated), signal_energy(test_data))

        assert np.allclose(rotated, np.asarray([[-3, -4, -5], [0, 1, 2]]))

    @staticmethod
    def test_common_rotation_2() -> None:
        # 45 degree rotation.
        rot = PolarizationRotation(np.pi / 4)

        test_data = np.asarray((np.arange(4), np.arange(4)), dtype=np.cdouble)

        rotated = rot(test_data)

        assert rotated.shape == test_data.shape
        assert rotated.dtype == test_data.dtype

        assert np.isclose(signal_energy(rotated), signal_energy(test_data))

        assert np.allclose(
            rotated, np.asarray((np.zeros(4), np.sqrt(2 * np.arange(4) ** 2)))
        )

    @staticmethod
    def test_no_rotation() -> None:
        # 0 degree rotation.
        rot = PolarizationRotation(0)

        test_data = np.arange(6, dtype=np.cdouble).reshape(2, 3) + 2j

        rotated = rot(test_data)

        assert rotated.shape == test_data.shape
        assert rotated.dtype == test_data.dtype

        assert np.isclose(signal_energy(rotated), signal_energy(test_data))

        assert np.allclose(rotated, test_data)

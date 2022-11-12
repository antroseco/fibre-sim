from math import ceil, log2
from typing import Sequence

import numpy as np
import pytest
from scipy.signal import convolve, lfilter
from utils import (
    calculate_awgn_ber_with_bpsk,
    calculate_awgn_ser_with_qam,
    is_power_of_2,
    signal_power,
    next_power_of_2,
    normalize_energy,
    overlap_save,
    signal_energy,
)


class TestCalculateAwgnBerWithBpsk:
    @staticmethod
    def test_monotonically_decreasing():
        x = np.linspace(1, 10, 20)
        y = calculate_awgn_ber_with_bpsk(x)

        assert np.all(np.diff(y) < 0)

    @staticmethod
    def test_known_values():
        assert np.allclose(
            calculate_awgn_ber_with_bpsk(np.asarray((4, 9))),
            (0.00233886749, 0.00001104524),
        )


class TestCalculateAwgnSerWithQam:
    @staticmethod
    def test_monotonically_decreasing():
        x = np.linspace(1, 10, 20)
        y = calculate_awgn_ser_with_qam(16, x)

        assert np.all(np.diff(y) < 0)

    @staticmethod
    def test_known_values():
        assert np.allclose(
            calculate_awgn_ser_with_qam(16, np.asarray((4, 9))),
            (0.11045740518, 0.01093553713),
        )


class TestNextPowerOf2:
    @staticmethod
    @pytest.mark.parametrize("value", (1, 2, 3, 4, 7, 8, 15, 16))
    def test_valid(value: int):
        assert next_power_of_2(value) == 2 ** (ceil(log2(value) - 1) + 1)

    @staticmethod
    @pytest.mark.parametrize("value", (-1, 0))
    def test_invalid(value: int):
        with pytest.raises(Exception):
            next_power_of_2(value)


class TestIsPowerOf2:
    @staticmethod
    @pytest.mark.parametrize("value", (1, 2, 4, 8, 1024))
    def test_valid(value: int):
        assert is_power_of_2(value)

    @staticmethod
    @pytest.mark.parametrize("value", (-1, 0, 3, 5, 1023))
    def test_invalid(value: int):
        assert not is_power_of_2(value)


class TestOverlapSave:
    @staticmethod
    @pytest.mark.parametrize("fir_length", (1, 2, 3, 4, 5, 6, 7, 8))
    @pytest.mark.parametrize("data_length", (8, 16, 1024, 4096))
    def test_random(fir_length: int, data_length: int):
        rng = np.random.default_rng()

        h = rng.normal(size=fir_length) + 1j * rng.normal(size=fir_length)
        x = rng.normal(size=data_length) + 1j * rng.normal(size=data_length)

        # Test default mode "same".
        result = overlap_save(h, x)
        expected = lfilter(h, 1, x, zi=None)

        assert result.size == np.size(expected)
        assert np.allclose(result, expected)

        # Test mode "full".
        result = overlap_save(h, x, full=True)
        expected = convolve(h, x)

        assert result.size == np.size(expected)

    @staticmethod
    @pytest.mark.parametrize(
        "fir,expected",
        (
            ((1, 2), (1, 2, 0, 2, 4, 0, 3, 6, 0, 4, 8, 0, 5, 10, 0, 0)),
            ((-1, 2, 1), (-1, 2, 1, -2, 4, 2, -3, 6, 3, -4, 8, 4, -5, 10, 5, 0)),
            ((-1, 2, 1, 2), (-1, 2, 1, 0, 4, 2, 1, 6, 3, 2, 8, 4, 3, 10, 5, 10)),
        ),
    )
    def test_simple(fir: Sequence[float], expected: Sequence[float]):
        h = np.asarray(fir)
        x = np.asarray((1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 0))

        result = overlap_save(h, x)
        assert np.allclose(result, expected)

    @staticmethod
    def test_invalid():
        with pytest.raises(Exception):
            overlap_save(np.arange(0), np.arange(1))

        with pytest.raises(Exception):
            overlap_save(np.arange(1), np.arange(0))


class TestEnergy:
    @staticmethod
    @pytest.mark.parametrize("amplitude", range(4))
    def test_signal_energy(amplitude: int):
        LENGTH = 16

        signal = np.full(LENGTH, amplitude)
        energy = signal_energy(signal)

        assert np.isfinite(energy)
        assert energy >= 0
        assert np.isclose(energy, LENGTH * amplitude**2)

    @staticmethod
    @pytest.mark.parametrize("amplitude", range(4))
    def test_mean_sample_energy(amplitude: int):
        LENGTH = 16

        signal = np.full(LENGTH, amplitude)
        energy = signal_power(signal)

        assert np.isfinite(energy)
        assert energy >= 0
        assert np.isclose(energy, amplitude**2)

    @staticmethod
    def test_normalize_energy():
        LENGTH = 128

        signal = np.full(LENGTH, 3 + 3j)
        normalized = normalize_energy(signal)

        assert normalized.dtype == signal.dtype
        assert normalized.ndim == signal.ndim
        assert normalized.size == signal.size
        assert np.all(np.isfinite(normalized))

        # Signal energy must be 1.
        assert np.isclose(signal_energy(normalized), 1)

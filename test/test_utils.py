from math import ceil, log2
from typing import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import convolve, lfilter

from utils import (
    bits_to_ints,
    calculate_awgn_ber_with_bpsk,
    calculate_awgn_ser_with_qam,
    for_each_polarization,
    has_one_polarization,
    has_two_polarizations,
    ints_to_bits,
    is_power_of_2,
    next_power_of_2,
    normalize_energy,
    normalize_power,
    optimum_overlap_save_frame_size,
    overlap_save,
    power_dbm_to_lin,
    signal_energy,
    signal_power,
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

    @staticmethod
    @pytest.mark.parametrize(
        "h_size,expected",
        (
            (4, 16),
            (5, 16),
            # M=8 is a special case: N=32 and N=64 are both optimal, should
            # return the smallest of the two.
            (8, 32),
            (60, 512),
            (150, 1024),
        ),
    )
    def test_optimum_overlap_save_frame_size(h_size: int, expected: int) -> None:
        assert optimum_overlap_save_frame_size(h_size) == expected


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

    @staticmethod
    def test_normalize_power():
        LENGTH = 128

        signal = np.random.randn(LENGTH) + 1j * np.random.randn(LENGTH)
        normalized = normalize_power(signal)

        assert normalized.dtype == signal.dtype
        assert normalized.ndim == signal.ndim
        assert normalized.size == signal.size
        assert np.all(np.isfinite(normalized))

        # Signal power must be 1.
        assert np.isclose(signal_power(normalized), 1)

    @staticmethod
    @pytest.mark.parametrize(
        "dBm,expected",
        ((-10, 0.0001), (-5, 0.00031622776602), (0, 0.001), (42, 15.848931925)),
    )
    def test_power_dbm_to_lin(dBm: float, expected: float):
        result = power_dbm_to_lin(dBm)

        assert np.isfinite(result)
        assert np.isclose(result, expected)


def test_ints_to_bits():
    # Test that bits are returned in the correct order (MSB first).
    assert np.all(ints_to_bits(np.asarray((6,)), 3) == [True, True, False])

    # Test different numbers of bits per int.
    assert np.all(
        ints_to_bits(np.asarray((2, 1)), 3) == [False, True, False, False, False, True]
    )
    assert np.all(ints_to_bits(np.asarray((2, 1)), 2) == [True, False, False, True])
    assert np.all(ints_to_bits(np.asarray((2, 1)), 1) == [False, True])


def test_bits_to_ints():
    bits = np.asarray((True, False, True, False))

    # Test that bits are interpreted correctly (first one is MSB).
    assert np.all(bits_to_ints(bits, 1) == [1, 0, 1, 0])
    assert np.all(bits_to_ints(bits, 2) == [2, 2])
    assert np.all(bits_to_ints(bits, 4) == [10])

    # Size of bits must be a multiple of bits_per_int.
    with pytest.raises(Exception):
        bits_to_ints(bits, 0)
    with pytest.raises(Exception):
        bits_to_ints(bits, 3)
    with pytest.raises(Exception):
        bits_to_ints(bits, 5)

    # bits can't be empty.
    with pytest.raises(Exception):
        bits_to_ints(np.asarray((), dtype=np.bool_), 2)


class TestForEachPolarization:
    @for_each_polarization
    def double(self, array: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(array)

        return array * 2

    def test_one_polarization(self) -> None:
        test_data = np.arange(4, dtype=np.cdouble) + 2j

        doubled = self.double(test_data)

        assert has_one_polarization(doubled)
        assert doubled.size == test_data.size
        assert doubled.dtype == test_data.dtype

        assert np.allclose(doubled, test_data * 2)

    def test_two_polarization(self) -> None:
        test_data = np.arange(8, dtype=np.cdouble).reshape(2, 4) + 2j

        doubled = self.double(test_data)

        assert has_two_polarizations(doubled)
        assert doubled.size == test_data.size
        assert doubled.dtype == test_data.dtype

        assert np.allclose(doubled, test_data * 2)

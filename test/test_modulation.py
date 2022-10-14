from math import floor

import numpy as np
import pytest
from modulation import DemodulatorBPSK, DemodulatorQPSK, ModulatorBPSK, ModulatorQPSK
from numpy.typing import NDArray


def array_to_bits(array: NDArray) -> NDArray[np.bool8]:
    assert array.ndim == 1

    # Determine how many bits are required to represent all values.
    # The call to max() protects against cases where the greatest value is 0.
    bit_count = floor(np.log2(max(array.max(), 1))) + 1

    # Place each element in its own row. The count argument in unpackbits()
    # doesn't do what we want it do otherwise.
    array = np.reshape(array, (array.size, 1)).astype(np.uint8)

    # Unpack each row individually.
    array = np.unpackbits(array, axis=1, count=bit_count, bitorder="little")

    # bitorder="little" returns the LSB first, so we need to use fliplr() to
    # bring the MSB to the front.
    # .ravel() is like .flatten() but it doesn't copy the array.
    return np.fliplr(array).ravel().astype(np.bool8)


class TestModulatorBPSK:
    modulator = ModulatorBPSK()

    def test_mapping(self):
        data = array_to_bits(np.arange(2))
        symbols = self.modulator(data)

        # Check dtype.
        assert symbols.dtype == np.cdouble

        # Length should be preserved.
        assert data.size == symbols.size

        # 0 must be mapped to 1.
        assert symbols[0] == 1

        # 1 must be mapped to -1.
        assert symbols[1] == -1


class TestDemodulatorBPSK:
    demodulator = DemodulatorBPSK()

    def test_demodulator(self):
        symbols = np.asarray((-1, -0.5, 0.5, 1), dtype=np.cdouble)
        data = self.demodulator(symbols)

        # Check dtype.
        assert data.dtype == np.bool8

        # Length should be preserved.
        assert symbols.size == data.size

        # Symbols less than 0 should be mapped to 1.
        assert data[0] == 1
        assert data[1] == 1

        # Symbols greater than 0 should be mapped to 0.
        assert data[2] == 0
        assert data[3] == 0

    def test_combined(self):
        LENGTH = 1024

        modulator = ModulatorBPSK()
        rng = np.random.default_rng()

        # Modulation should be reversible.
        data = rng.integers(0, 1, endpoint=True, size=LENGTH, dtype=np.bool8)
        assert np.all(self.demodulator(modulator(data)) == data)

        # Small amounts of noise should not introduce any errors.
        noise = rng.uniform(-0.5, 0.5, size=LENGTH)
        assert np.all(self.demodulator(modulator(data) + noise) == data)


class TestModulatorQPSK:
    modulator = ModulatorQPSK()

    def test_mapping(self):
        data = array_to_bits(np.arange(4))
        symbols = self.modulator(data)

        # Check dtype.
        assert symbols.dtype == np.cdouble

        # Length should be halved (2 bits per symbol).
        assert data.size == 2 * symbols.size

        # All symbols must have unit energy.
        assert np.allclose(np.abs(symbols), 1)

        # Un-normalize energy to make the comparisons below easier.
        symbols *= np.sqrt(2)

        # Check constellation (diagram in the class definition).
        assert symbols[0] == 1 + 1j
        assert symbols[1] == 1 - 1j
        assert symbols[2] == -1 + 1j
        assert symbols[3] == -1 - 1j


class TestDemodulatorQPSK:
    demodulator = DemodulatorQPSK()

    def test_demodulator(self):
        symbols = np.asarray((1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j), dtype=np.cdouble)
        data = self.demodulator(symbols)

        # Check dtype.
        assert data.dtype == np.bool8

        # Length should be halved (2 bits per symbol).
        assert 2 * symbols.size == data.size

        # Check that symbols have been demodulated correctly.
        ints = np.packbits(np.reshape(data, (4, 2)), axis=1, bitorder="little")
        assert ints[0] == 0b00
        assert ints[1] == 0b01
        assert ints[2] == 0b10
        assert ints[3] == 0b11

    def test_combined(self):
        modulator = ModulatorQPSK()
        rng = np.random.default_rng()

        BIT_LENGTH = 1024
        SYM_LENGTH = BIT_LENGTH // modulator.bits_per_symbol

        # Modulation should be reversible.
        data = rng.integers(0, 1, endpoint=True, size=BIT_LENGTH, dtype=np.bool8)
        assert np.all(self.demodulator(modulator(data)) == data)

        # Small amounts of noise should not introduce any errors.
        noise_r = rng.uniform(-0.5, 0.5, size=SYM_LENGTH)
        noise_i = rng.uniform(-0.5, 0.5, size=SYM_LENGTH) * 1j
        assert np.all(self.demodulator(modulator(data) + noise_r + noise_i) == data)

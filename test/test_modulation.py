from itertools import product

import numpy as np
import pytest

from modulation import (
    Demodulator16QAM,
    DemodulatorBPSK,
    DemodulatorQPSK,
    Modulator16QAM,
    ModulatorBPSK,
    ModulatorQPSK,
)
from utils import bits_to_ints, ints_to_bits


class TestModulatorBPSK:
    modulator = ModulatorBPSK()

    def test_mapping(self):
        data = ints_to_bits(np.arange(2))
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
        assert data.dtype == np.bool_

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
        data = rng.integers(0, 1, endpoint=True, size=LENGTH, dtype=np.bool_)
        assert np.all(self.demodulator(modulator(data)) == data)

        # Small amounts of noise should not introduce any errors.
        noise = rng.uniform(-0.5, 0.5, size=LENGTH)
        assert np.all(self.demodulator(modulator(data) + noise) == data)


class TestModulatorQPSK:
    modulator = ModulatorQPSK()

    def test_mapping(self):
        data = ints_to_bits(np.arange(4))
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

    def test_odd_bit_lengths(self):
        # Only multiples of 2 are acceptable.
        assert self.modulator.bits_per_symbol == 2

        with pytest.raises(Exception):
            self.modulator(np.zeros(1, dtype=np.bool_))

        with pytest.raises(Exception):
            self.modulator(np.zeros(1001, dtype=np.bool_))


class TestDemodulatorQPSK:
    demodulator = DemodulatorQPSK()

    def test_demodulator(self):
        symbols = np.asarray((1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j), dtype=np.cdouble)
        data = self.demodulator(symbols)

        # Check dtype.
        assert data.dtype == np.bool_

        # Length should have doubled (2 bits per symbol).
        assert 2 * symbols.size == data.size

        # Check that symbols have been demodulated correctly.
        ints = bits_to_ints(data, self.demodulator.bits_per_symbol)
        assert ints[0] == 0b00
        assert ints[1] == 0b10
        assert ints[2] == 0b01
        assert ints[3] == 0b11

    def test_combined(self):
        modulator = ModulatorQPSK()
        rng = np.random.default_rng()

        BIT_LENGTH = 1024
        SYM_LENGTH = BIT_LENGTH // modulator.bits_per_symbol

        # Modulation should be reversible.
        data = rng.integers(0, 1, endpoint=True, size=BIT_LENGTH, dtype=np.bool_)
        assert np.all(self.demodulator(modulator(data)) == data)

        # Small amounts of noise should not introduce any errors.
        noise_r = rng.uniform(-0.5, 0.5, size=SYM_LENGTH)
        noise_i = rng.uniform(-0.5, 0.5, size=SYM_LENGTH) * 1j
        assert np.all(self.demodulator(modulator(data) + noise_r + noise_i) == data)


class TestModulator16QAM:
    modulator = Modulator16QAM()

    def test_impl(self):
        msbs = np.asarray((0, 0, 1, 1), dtype=np.bool_)
        lsbs = np.asarray((0, 1, 0, 1), dtype=np.bool_)

        offsets = self.modulator.impl(msbs, lsbs)

        # TODO explanation.
        assert offsets[0] == 0
        assert offsets[1] == 1
        assert offsets[2] == 3
        assert offsets[3] == 2

    def test_mapping(self):
        data = ints_to_bits(np.arange(16))
        symbols = self.modulator(data)

        # Check dtype.
        assert symbols.dtype == np.cdouble

        # Length should be a quarter (4 bits per symbol).
        assert data.size == 4 * symbols.size

        # Mean energy should be 1, although not all symbols have unit energy.
        assert np.isclose(np.mean(np.abs(symbols) ** 2), 1)

        # Un-normalize energy to make the comparisons below easier.
        symbols *= np.sqrt(10)

        # Check constellation (diagram in the class definition).
        assert symbols[0] == -3 + 3j
        assert symbols[1] == -1 + 3j
        assert symbols[2] == 3 + 3j
        assert symbols[3] == 1 + 3j
        assert symbols[4] == -3 + 1j
        assert symbols[5] == -1 + 1j
        assert symbols[6] == 3 + 1j
        assert symbols[7] == 1 + 1j
        assert symbols[8] == -3 - 3j
        assert symbols[9] == -1 - 3j
        assert symbols[10] == 3 - 3j
        assert symbols[11] == 1 - 3j
        assert symbols[12] == -3 - 1j
        assert symbols[13] == -1 - 1j
        assert symbols[14] == 3 - 1j
        assert symbols[15] == 1 - 1j

    def test_odd_bit_lengths(self):
        # Only multiples of 4 are acceptable.
        assert self.modulator.bits_per_symbol == 4

        with pytest.raises(Exception):
            self.modulator(np.zeros(1, dtype=np.bool_))

        with pytest.raises(Exception):
            self.modulator(np.zeros(2, dtype=np.bool_))

        with pytest.raises(Exception):
            self.modulator(np.zeros(3, dtype=np.bool_))

        with pytest.raises(Exception):
            self.modulator(np.zeros(1001, dtype=np.bool_))


class TestDemodulator16QAM:
    demodulator = Demodulator16QAM()

    def test_demodulator(self):
        # Enumerates the entire constellation column-by-column, starting with
        # leftmost column (in-phase = -3) and moving from bottom to top
        # (quadrature from -3 to +3).
        # FIXME replace magic number.
        symbol_basis = (-3, -1, 1, 3)
        symbols = np.fromiter(
            map(lambda r_i: r_i[0] + r_i[1] * 1j, product(symbol_basis, symbol_basis)),
            dtype=np.cdouble,
        ) / np.sqrt(10)

        data = self.demodulator(symbols)

        # Check dtype.
        assert data.dtype == np.bool_

        # Length should have quadrupled (4 bits per symbol).
        assert 4 * symbols.size == data.size

        # Check that symbols have been demodulated correctly.
        ints = bits_to_ints(data, self.demodulator.bits_per_symbol)
        assert ints[0] == 0b1000
        assert ints[1] == 0b1100
        assert ints[2] == 0b0100
        assert ints[3] == 0b0000
        assert ints[4] == 0b1001
        assert ints[5] == 0b1101
        assert ints[6] == 0b0101
        assert ints[7] == 0b0001
        assert ints[8] == 0b1011
        assert ints[9] == 0b1111
        assert ints[10] == 0b0111
        assert ints[11] == 0b0011
        assert ints[12] == 0b1010
        assert ints[13] == 0b1110
        assert ints[14] == 0b0110
        assert ints[15] == 0b0010

    def test_combined(self):
        modulator = Modulator16QAM()
        rng = np.random.default_rng()

        BIT_LENGTH = 1024
        SYM_LENGTH = BIT_LENGTH // modulator.bits_per_symbol

        # Modulation should be reversible.
        data = rng.integers(0, 1, endpoint=True, size=BIT_LENGTH, dtype=np.bool_)
        assert np.all(self.demodulator(modulator(data)) == data)

        # Small amounts of noise should not introduce any errors.
        noise_r = rng.uniform(-0.2, 0.2, size=SYM_LENGTH)
        noise_i = rng.uniform(-0.2, 0.2, size=SYM_LENGTH) * 1j
        assert np.all(self.demodulator(modulator(data) + noise_r + noise_i) == data)

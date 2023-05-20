from itertools import product

import numpy as np
import pytest
import scipy.io

from modulation import (
    AlamoutiEncoder,
    Demodulator16QAM,
    DemodulatorBPSK,
    DemodulatorDQPSK,
    DemodulatorQPSK,
    Modulator16QAM,
    ModulatorBPSK,
    ModulatorQPSK,
)
from utils import (
    bits_to_ints,
    has_two_polarizations,
    ints_to_bits,
    normalize_power,
    row_size,
)


class TestModulatorBPSK:
    modulator = ModulatorBPSK()

    def test_mapping(self):
        data = ints_to_bits(np.arange(2), 1)
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
        data = ints_to_bits(np.arange(4), 2)
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
        assert symbols[1] == -1 + 1j
        assert symbols[2] == 1 - 1j
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
        assert ints[1] == 0b01
        assert ints[2] == 0b10
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


class TestDemodulatorDQPSK:
    demodulator = DemodulatorDQPSK()

    def test_gray_to_binary(self) -> None:
        gray = np.asarray((0b00, 0b01, 0b11, 0b10), dtype=np.uint8)

        binary = self.demodulator.gray_to_binary(gray)

        assert binary.dtype == np.uint8
        assert binary.size == gray.size
        assert np.all(binary == np.arange(4))

    def test_demodulator(self) -> None:
        # fmt: off
        data = np.asarray(
            (
                2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 2, 2, 2, 3, 1, 3, 3, 3, 2, 1,
                3, 3, 1, 2, 3, 2, 2, 1, 1, 2, 0
            ),
            dtype=np.uint8,
        )
        expected = np.asarray(
            (
                3, 1, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0, 0, 3, 3, 1, 0, 0, 1, 2,
                1, 0, 3, 2, 3, 1, 0, 2, 0, 2, 1
            ),
            dtype=np.uint8,
        )
        # fmt: on

        symbols = ModulatorQPSK()(ints_to_bits(data, 2))

        assert symbols.dtype == np.cdouble
        assert symbols.size == data.size

        decoded_bits = self.demodulator(symbols)

        assert decoded_bits.dtype == np.bool_
        assert decoded_bits.size == 2 * symbols.size

        decoded = bits_to_ints(decoded_bits, 2)

        assert decoded.size == expected.size
        assert np.all(decoded == expected)


class TestModulator16QAM:
    modulator = Modulator16QAM()

    def test_impl(self):
        msbs = np.asarray((0, 0, 1, 1), dtype=np.bool_)
        lsbs = np.asarray((0, 1, 0, 1), dtype=np.bool_)

        offsets = self.modulator.impl(msbs, lsbs)

        # The constellation is Gray-coded, i.e. the sequence is 00, 01, 11, 10.
        # The in-phase and quadrature components each carry a 2-bit Gray-coded
        # sequence. This test ensures that we can convert from binary to the
        # Gray code used.
        assert offsets[0] == 0
        assert offsets[1] == 1
        assert offsets[2] == 3
        assert offsets[3] == 2

    def test_mapping(self):
        data = ints_to_bits(np.arange(16), 4)
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
        assert symbols[1] == -3 + 1j
        assert symbols[2] == -3 - 3j
        assert symbols[3] == -3 - 1j
        assert symbols[4] == -1 + 3j
        assert symbols[5] == -1 + 1j
        assert symbols[6] == -1 - 3j
        assert symbols[7] == -1 - 1j
        assert symbols[8] == 3 + 3j
        assert symbols[9] == 3 + 1j
        assert symbols[10] == 3 - 3j
        assert symbols[11] == 3 - 1j
        assert symbols[12] == 1 + 3j
        assert symbols[13] == 1 + 1j
        assert symbols[14] == 1 - 3j
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

    def test_against_matlab(self) -> None:
        # Test against MATLAB's qammod(x, 16, 'gray'). There is more than one
        # possible constellation, but it's nice to be compatible.
        expected = normalize_power(
            scipy.io.loadmat("test/matlab_16qam_gray.mat")["symgray"].ravel()
        )

        data = ints_to_bits(np.arange(16), 4)
        result = self.modulator(data)

        assert result.dtype == expected.dtype
        assert result.size == expected.size
        assert np.allclose(result, expected)


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
        assert ints[0] == 0b0010
        assert ints[1] == 0b0011
        assert ints[2] == 0b0001
        assert ints[3] == 0b0000
        assert ints[4] == 0b0110
        assert ints[5] == 0b0111
        assert ints[6] == 0b0101
        assert ints[7] == 0b0100
        assert ints[8] == 0b1110
        assert ints[9] == 0b1111
        assert ints[10] == 0b1101
        assert ints[11] == 0b1100
        assert ints[12] == 0b1010
        assert ints[13] == 0b1011
        assert ints[14] == 0b1001
        assert ints[15] == 0b1000

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

    def test_against_matlab(self) -> None:
        # Test against MATLAB's qammod(x, 16, 'gray'). There is more than one
        # possible constellation, but it's nice to be compatible.
        symbols = scipy.io.loadmat("test/matlab_16qam_gray.mat")["symgray"].ravel()
        result = self.demodulator(symbols)

        expected = ints_to_bits(np.arange(16), 4)

        assert result.dtype == expected.dtype
        assert result.size == expected.size
        assert np.allclose(result, expected)


class TestAlamoutiEncoder:
    encoder = AlamoutiEncoder()

    def test_encoder(self) -> None:
        data = np.arange(1, 5) + 1j * np.arange(1, 5)
        expected_y = (data[1], np.conj(data[0]), data[3], np.conj(data[2]))
        expected_x = (data[0], -np.conj(data[1]), data[2], -np.conj(data[3]))

        encoded = self.encoder(data)

        assert has_two_polarizations(encoded)
        assert encoded.size == len(expected_y) + len(expected_x)
        assert row_size(encoded) == len(expected_x) == len(expected_y)
        assert encoded.dtype == data.dtype

        encoded_x, encoded_y = encoded

        assert np.allclose(encoded_x, expected_x)
        assert np.allclose(encoded_y, expected_y)

    def test_against_lab_data(self) -> None:
        data = scipy.io.loadmat("test/AC_data.mat")

        # Original 16-QAM symbols.
        s_qam = np.ravel(data["s_qam"])

        # Odd and even symbols (1-indexed). Not very interesting.
        so = np.ravel(data["so"])
        se = np.ravel(data["se"])

        assert s_qam.size == 2 * so.size
        assert s_qam.size == 2 * se.size

        odd = s_qam[0::2]
        even = s_qam[1::2]

        assert np.allclose(odd, so)
        assert np.allclose(even, se)

        # These signals are prefixed with 2048 QAM (for synchronization).
        # Following those, are the two Alamouti-coded signals (one for each
        # polarization)...
        sx2 = np.ravel(data["sx2"])[2048:]
        sy2 = np.ravel(data["sy2"])[2048:]

        assert sx2.size == s_qam.size
        assert sy2.size == s_qam.size

        # ...which we can test our encoder against.
        px, py = self.encoder(s_qam)

        assert np.allclose(px, sx2)
        assert np.allclose(py, sy2)

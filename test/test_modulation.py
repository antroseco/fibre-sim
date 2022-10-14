import numpy as np
import pytest
from modulation import DemodulatorBPSK, DemodulatorQPSK, ModulatorBPSK, ModulatorQPSK


class TestModulatorBPSK:
    modulator = ModulatorBPSK()

    def test_bounds_checking(self):
        with pytest.raises(Exception):
            self.modulator(np.asarray(-1))
        with pytest.raises(Exception):
            self.modulator(np.asarray(2))

        # Only 0 and 1 are allowed.
        self.modulator(np.asarray((0, 1)))

    def test_mapping(self):
        data = np.arange(2, dtype=np.uint8)
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
        assert data.dtype == np.uint8

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
        data = rng.integers(0, 1, endpoint=True, size=LENGTH)
        assert np.all(self.demodulator(modulator(data)) == data)

        # Small amounts of noise should not introduce any errors.
        noise = rng.uniform(-0.5, 0.5, size=LENGTH)
        assert np.all(self.demodulator(modulator(data) + noise) == data)


class TestModulatorQPSK:
    modulator = ModulatorQPSK()

    def test_bounds_checking(self):
        with pytest.raises(Exception):
            self.modulator(np.asarray(-1))
        with pytest.raises(Exception):
            self.modulator(np.asarray(4))

        # Only 2 bit numbers are allowed.
        self.modulator(np.asarray((0, 1, 2, 3)))

    def test_mapping(self):
        data = np.arange(4, dtype=np.uint8)
        symbols = self.modulator(data)

        # Check dtype.
        assert symbols.dtype == np.cdouble

        # Length should be preserved.
        assert data.size == symbols.size

        # All symbols must have unit energy.
        assert np.allclose(np.abs(symbols), 1)

        # Un-normalize energy to make the comparisons below easier.
        symbols *= np.sqrt(2)

        # Check constellation.
        #         Q
        #   01    |    00
        #         |
        #  ---------------I
        #         |
        #   11    |    10
        #         |
        assert symbols[0] == 1 + 1j
        assert symbols[1] == -1 + 1j
        assert symbols[2] == 1 - 1j
        assert symbols[3] == -1 - 1j


class TestDemodulatorQPSK:
    demodulator = DemodulatorQPSK()

    def test_demodulator(self):
        symbols = np.asarray((1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j), dtype=np.cdouble)
        data = self.demodulator(symbols)

        # Check dtype.
        assert data.dtype == np.uint8

        # Length should be preserved.
        assert symbols.size == data.size

        # Check that symbols have been demodulated correctly.
        assert data[0] == 0b00
        assert data[1] == 0b01
        assert data[2] == 0b10
        assert data[3] == 0b11

    def test_combined(self):
        LENGTH = 1024

        modulator = ModulatorQPSK()
        rng = np.random.default_rng()

        # Modulation should be reversible.
        data = rng.integers(0, 3, endpoint=True, size=LENGTH)
        assert np.all(self.demodulator(modulator(data)) == data)

        # Small amounts of noise should not introduce any errors.
        noise_r = rng.uniform(-0.5, 0.5, size=LENGTH)
        noise_i = rng.uniform(-0.5, 0.5, size=LENGTH) * 1j
        assert np.all(self.demodulator(modulator(data) + noise_r + noise_i) == data)

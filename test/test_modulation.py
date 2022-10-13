import pytest
import numpy as np
from modulation import DemodulatorBPSK, ModulatorBPSK


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
        data = np.asarray((0, 1))
        symbols = self.modulator(data)

        # Length should be preserved.
        assert data.size == symbols.size

        # 0 must be mapped to 1
        assert symbols[0] == 1

        # 1 must be mapped to -1
        assert symbols[1] == -1


class TestDemodulatorBPSK:
    demodulator = DemodulatorBPSK()

    def test_demodulator(self):
        symbols = np.asarray((-1, -0.5, 0.5, 1))
        data = self.demodulator(symbols)

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

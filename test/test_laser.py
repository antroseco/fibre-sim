import numpy as np

from laser import ContinuousWaveLaser
from utils import power_dbm_to_lin, signal_power


class TestContinuousWaveLaser:
    @staticmethod
    def test_power():
        POWER_dBm = 20
        POWER = power_dbm_to_lin(POWER_dBm)
        LENGTH = 2**8

        laser = ContinuousWaveLaser(POWER_dBm)
        result = laser(LENGTH)

        assert result.ndim == 1
        assert result.size == LENGTH
        assert result.dtype == np.cdouble

        # All samples must have the same amplitude.
        assert np.all(result[0] == result)

        # Samples shouldn't have phase.
        assert np.allclose(np.angle(result), 0)

        # Power must be set correctly.
        assert np.isclose(signal_power(result), POWER)

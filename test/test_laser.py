import numpy as np

from laser import ContinuousWaveLaser, NoisyLaser
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


class TestNoisyLaser:
    POWER_dBm = 20
    POWER = power_dbm_to_lin(POWER_dBm)
    SAMPLING_RATE = 1e12

    laser = NoisyLaser(POWER_dBm, sampling_rate=SAMPLING_RATE)

    def test_power(self):
        LENGTH = 2**8

        result = self.laser(LENGTH)

        assert result.ndim == 1
        assert result.size == LENGTH
        assert result.dtype == np.cdouble

        # All samples must have the same amplitude.
        assert np.allclose(np.abs(result[0]), np.abs(result))

        # Power must be set correctly.
        assert np.isclose(signal_power(result), self.POWER)

    def test_wiener_process(self):
        LENGTH = 2**6

        phase_changes = []

        for _ in range(8000):
            result = self.laser(LENGTH)
            # Variance is very small, so we don't expect the angle to wrap
            # around.
            phase_changes.append(np.angle(result[-1]) - np.angle(result[0]))

        # Check mean.
        assert np.isclose(np.mean(phase_changes), 0, atol=1e-3)

        # Check variance.
        expected_var = 2 * np.pi * self.laser.LINEWIDTH * LENGTH / self.SAMPLING_RATE
        assert np.isclose(np.var(phase_changes), expected_var, rtol=0.05)

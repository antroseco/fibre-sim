from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from utils import power_dbm_to_lin


class Laser(ABC):
    def __init__(self, power_dbm: float) -> None:
        super().__init__()

        self.power = power_dbm_to_lin(power_dbm)

    @abstractmethod
    def __call__(self, size: int) -> NDArray[np.cdouble]:
        pass


class ContinuousWaveLaser(Laser):
    def __call__(self, size: int) -> NDArray[np.cdouble]:
        # Continuous wave of constant amplitude and phase.
        return np.full(size, np.sqrt(self.power), dtype=np.cdouble)


class NoisyLaser(ContinuousWaveLaser):
    # Typical in coherent systems, according to Digital Coherent Optical
    # Systems.
    LINEWIDTH = 100e3  # Hz

    def __init__(self, power_dbm: float, sampling_rate: float) -> None:
        super().__init__(power_dbm)

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

        self.rng = np.random.default_rng()

        # Aids testing and debugging.
        self.last_noise: Optional[NDArray[np.float64]] = None

    def __call__(self, size: int) -> NDArray[np.cdouble]:
        amplitudes = super().__call__(size)

        noise_step_var = 2 * np.pi * self.LINEWIDTH * self.sampling_interval

        # Note that scale is the standard deviation.
        noise_steps = self.rng.normal(loc=0, scale=np.sqrt(noise_step_var), size=size)

        # Modelled as a Wiener process.
        self.last_noise = np.cumsum(noise_steps)

        return amplitudes * np.exp(1j * self.last_noise)

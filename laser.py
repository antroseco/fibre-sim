from abc import ABC, abstractmethod

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

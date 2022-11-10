from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Laser(ABC):
    @abstractmethod
    def __call__(self, size: int) -> NDArray[np.cdouble]:
        pass


class ContinuousWaveLaser(Laser):
    def __call__(self, size: int) -> NDArray[np.cdouble]:
        # Continuous wave of constant amplitude and phase.
        return np.ones(size, np.cdouble)

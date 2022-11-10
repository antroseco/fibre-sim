from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Planck, elementary_charge, speed_of_light

from utils import Component


class OpticalFrontEnd(Component):
    input_type = "cd electric field"
    output_type = "cd symbols"

    @cached_property
    def responsivity(self) -> float:
        # TODO what is a reasonable value for the quantum efficiency?
        efficiency = 0.6
        return (
            efficiency * elementary_charge * self.WAVELENGTH / (Planck * speed_of_light)
        )

    def __call__(self, Efields: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert Efields.ndim == 1
        assert Efields.size > 0

        # Assuming an ideal local oscillator with unit amplitude, and with
        # homodyne detection to maintain the baseband represtation, we just have
        # to multiply by the responsivity to get the photocurrent.
        # TODO implement intradyne detection.
        # TODO what is a reasonable local oscillator amplitude?
        return Efields * self.responsivity

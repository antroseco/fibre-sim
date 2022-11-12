from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Boltzmann, Planck, elementary_charge, speed_of_light

from utils import Component, power_dbm_to_lin, signal_power


class OpticalFrontEnd(Component):
    input_type = "cd electric field"
    output_type = "cd symbols"

    lo_power = power_dbm_to_lin(10)  # 10 dBm is the max for Class 1 lasers.
    lo_amplitude = np.sqrt(2 * lo_power)  # Signal power to peak amplitude.

    @cached_property
    def responsivity(self) -> float:
        # Responsivity is around 0.688 A/W. This seems like a reasonable value;
        # photodetectors with greater responsivities have been demonstrated
        # (e.g. Young-Ho Ko, Joong-Seon Choe, Won Seok Han, Seo-Young Lee,
        # Young-Tak Han, Hyun-Do Jung, Chun Ju Youn, Jong-Hoi Kim, and Yongsoon
        # Baek, "High-speed waveguide photodetector for 64 Gbaud coherent
        # receiver," Opt. Lett. 43, 579-582 (2018) measured 0.73 A/W).
        efficiency = 0.55
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
        return self.responsivity * self.lo_amplitude * Efields


class NoisyOpticalFrontEnd(OpticalFrontEnd):
    def __init__(self, sampling_rate: float, rx_power_dbm: float) -> None:
        super().__init__()

        assert sampling_rate > 0
        self.sampling_rate = sampling_rate

        self.rx_power = power_dbm_to_lin(rx_power_dbm)
        self.rng = np.random.default_rng()

    def __call__(self, Efields: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Set received signal power to that specified.
        # TODO remove this once we have proper amplitudes upstream.
        Efields *= np.sqrt(self.rx_power / signal_power(Efields))
        assert np.isclose(signal_power(Efields), self.rx_power)

        # Noise-less current.
        current = super().__call__(Efields)

        # σ2 = 2*e*r*B_N where r is the photocurrent (approximately R*P_LO).
        # Take the noise bandwidth as the Nyquist frequency.
        B_N = self.sampling_rate / 2
        shot_noise_stdev = np.sqrt(
            2 * elementary_charge * self.responsivity * self.lo_power * B_N
        )
        shot_noise_r = self.rng.normal(0, shot_noise_stdev, size=current.size)
        shot_noise_i = self.rng.normal(0, shot_noise_stdev, size=current.size)

        current += shot_noise_r
        current += 1j * shot_noise_i

        # Thermal noise has σ2 = 4*k_B*T*B_N/R_L. FIXME what should R_L be?
        thermal_noise_stdev = np.sqrt(4 * Boltzmann * 293 * B_N)
        thermal_noise_r = self.rng.normal(0, thermal_noise_stdev, size=current.size)
        thermal_noise_i = self.rng.normal(0, thermal_noise_stdev, size=current.size)

        current += thermal_noise_r
        current += 1j * thermal_noise_i

        return current

from functools import cached_property
from typing import Optional, Type

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Boltzmann, Planck, elementary_charge, speed_of_light

from laser import NoisyLaser
from utils import (
    Component,
    Signal,
    has_one_polarization,
    has_up_to_two_polarizations,
    power_dbm_to_lin,
)


class OpticalFrontEnd(Component):
    lo_power = power_dbm_to_lin(10)  # 10 dBm is the max for Class 1 lasers.
    lo_amplitude = np.sqrt(2 * lo_power)  # Signal power to peak amplitude.

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.OPTICAL, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

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
        # These are simple scalar multiplications.
        assert has_up_to_two_polarizations(Efields)

        # Assuming an ideal local oscillator with unit amplitude, and with
        # homodyne detection to maintain the baseband represtation, we just have
        # to multiply by the responsivity to get the photocurrent.
        return self.responsivity * self.lo_amplitude * Efields


class HeterodyneFrontEnd(OpticalFrontEnd):
    def __init__(
        self, if_ghz: float, sampling_rate: float, linewidth: float = 200e3
    ) -> None:
        super().__init__()

        assert if_ghz > 0
        self.if_omega = 2 * np.pi * if_ghz * 1e9

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

        self.laser = NoisyLaser(0, sampling_rate, linewidth)

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.float64, None

    @property
    def last_noise(self) -> Optional[NDArray[np.float64]]:
        return self.laser.last_noise

    def __call__(self, Efields: NDArray[np.cdouble]) -> NDArray[np.float64]:
        assert has_one_polarization(Efields)

        if_term = (self.if_omega * self.sampling_interval) * np.arange(Efields.size)

        self.laser.sample_phase_noise(Efields.size)
        assert self.last_noise is not None

        return (2 * self.responsivity * self.lo_amplitude) * np.real(
            Efields * np.exp(1j * (if_term - self.last_noise))
        )


class Digital90degHybrid(Component):
    def __init__(self, if_ghz: float, sampling_rate: float) -> None:
        super().__init__()

        assert if_ghz > 0
        self.if_omega = 2 * np.pi * if_ghz * 1e9

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(symbols)

        # LO, but conjugated.
        if_term = (self.if_omega * self.sampling_interval) * np.arange(symbols.size)
        loC = np.exp(-1j * if_term)

        # Conveniently, the in-phase component is the real part, and the
        # quadrature component is the imaginary part.
        return symbols * loC


class NoisyOpticalFrontEnd(OpticalFrontEnd):
    def __init__(self, sampling_rate: float) -> None:
        super().__init__()

        assert sampling_rate > 0
        self.sampling_rate = sampling_rate

        self.rng = np.random.default_rng()

    def __call__(self, Efields: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(Efields)

        # 50 Ω load resistor (common value).
        R_LOAD = 50

        # Noise-less current.
        current = super().__call__(Efields)

        # Take the noise bandwidth as the Nyquist frequency.
        B_N = self.sampling_rate / 2

        # σ2 = 2*e*r*B_N where r is the photocurrent (approximately R*P_LO).
        shot_noise_var = 2 * elementary_charge * self.responsivity * self.lo_power * B_N

        # Thermal noise has σ2 = 4*k_B*T*B_N/R_L.
        thermal_noise_var = 4 * Boltzmann * 293 * B_N / R_LOAD

        # Shot noise and thermal noise are independent, so their variances add.
        noise_stdev = np.sqrt(shot_noise_var + thermal_noise_var)

        current += self.rng.normal(0, noise_stdev, size=current.size)
        current += self.rng.normal(0, noise_stdev, size=current.size) * 1j

        # Convert current to voltage.
        return current * R_LOAD

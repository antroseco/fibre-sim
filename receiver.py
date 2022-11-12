from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Planck, elementary_charge, speed_of_light

from utils import Component, signal_energy, power_dbm_to_lin


class OpticalFrontEnd(Component):
    input_type = "cd electric field"
    output_type = "cd symbols"

    lo_power = power_dbm_to_lin(10)  # 10 dBm is the max for Class 1 lasers.
    lo_amplitude = np.sqrt(lo_power)  # Signal power is the mean sample squared.

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

        print(rx_power_dbm)
        self.rx_power = power_dbm_to_lin(rx_power_dbm)

    @classmethod
    def energy_to_photons(cls, energies: NDArray[np.float64]) -> NDArray[np.float64]:
        # Divide each sample energy by hf.
        return cls.WAVELENGTH / (Planck * speed_of_light) * energies

    @classmethod
    def photons_to_current(cls, photons: NDArray[np.int64]) -> NDArray[np.float64]:
        # Multiply each sample by hf.
        return Planck * speed_of_light / cls.WAVELENGTH * photons

    def photodiode_incident_energy(
        self, A_r: NDArray[np.float64], phi_r: NDArray[np.float64], lo_phase: float
    ) -> NDArray[np.float64]:
        energies = (
            A_r**2
            + self.lo_amplitude**2
            + 2 * A_r * self.lo_amplitude * np.cos(phi_r - lo_phase)
        )

        return energies / 4

    def photodiode_current(
        self, A_r: NDArray[np.float64], phi_r: NDArray[np.float64], lo_phase: float
    ) -> NDArray[np.float64]:
        energies = self.photodiode_incident_energy(A_r, phi_r, lo_phase)
        photons = self.energy_to_photons(self.responsivity * energies)

        # FIXME switch to a normal distribution and stop converting to and from
        # photons.
        noisy_photons = np.random.poisson(photons)
        currents = self.photons_to_current(noisy_photons)

        # Calculate SNR.
        signal = signal_energy(photons)
        noise = signal_energy(photons - noisy_photons)
        print("SNR", 10 * np.log10(signal / noise), "dB")

        return currents

    def __call__(self, Efields: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Set received signal power to that specified.
        # TODO remove this once we have proper amplitudes upstream.
        target_energy = self.rx_power * Efields.size / self.sampling_rate
        Efields *= np.sqrt(target_energy / signal_energy(Efields))

        A_r = np.abs(Efields)
        phi_r: NDArray[np.float64] = np.angle(Efields)  # type: ignore

        i1 = self.photodiode_current(A_r, phi_r, 0)
        i2 = self.photodiode_current(A_r, phi_r, np.pi)
        i3 = self.photodiode_current(A_r, phi_r, np.pi / 2)
        i4 = self.photodiode_current(A_r, phi_r, np.pi * 3 / 2)

        # Compute balanced photodetectors and pack into complex numbers.
        i = np.empty_like(Efields, np.cdouble)
        i.real = i1 - i2  # In-phase.
        i.imag = i3 - i4  # Quadrature.

        return i

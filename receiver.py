from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Planck, elementary_charge, speed_of_light

from utils import Component, power_dbm_to_lin, signal_power


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

        self.rx_power = power_dbm_to_lin(rx_power_dbm)

    def power_to_photons(self, powers: NDArray[np.float64]) -> NDArray[np.float64]:
        # powers is an array of instantaneous powers (i.e. the samples squared).
        # Need to multiply each instantaneous power by the sampling interval to
        # recover the sample's energy, before converting to photons.
        return self.WAVELENGTH / (self.sampling_rate * Planck * speed_of_light) * powers

    def photons_to_power(self, photons: NDArray[np.int64]) -> NDArray[np.float64]:
        # Multiply each sample by hf, and divide by the sampling interval to
        # convert energy to power.
        return self.sampling_rate * Planck * speed_of_light / self.WAVELENGTH * photons

    def photodiode_incident_power(
        self, A_r: NDArray[np.float64], phi_r: NDArray[np.float64], lo_phase: float
    ) -> NDArray[np.float64]:
        powers = (
            A_r**2
            + self.lo_amplitude**2
            + 2 * A_r * self.lo_amplitude * np.cos(phi_r - lo_phase)
        )

        return powers / 4

    def photodiode_current(
        self, A_r: NDArray[np.float64], phi_r: NDArray[np.float64], lo_phase: float
    ) -> NDArray[np.float64]:
        powers = self.photodiode_incident_power(A_r, phi_r, lo_phase)
        photons = self.power_to_photons(powers)
        assert np.isclose(self.power_to_photons(self.lo_amplitude**2), 97536)

        # FIXME switch to a normal distribution and stop converting to and from
        # photons.
        noisy_photons = np.random.poisson(photons)
        currents = self.responsivity * self.photons_to_power(noisy_photons)

        # print(f"{photons[:8]=}")
        # print(f"{noisy_photons[:8]=}")
        # Calculate SNR.
        signal = signal_power(photons)
        noise = signal_power(photons - noisy_photons)
        print(
            np.mean(photons),
            "photons, SNR",
            10 * np.log10(signal / noise),
            "dB",
            signal / noise,
            "lin",
        )
        noise = np.var(currents - self.responsivity * self.photons_to_power(photons))
        print(
            "var",
            noise,
            "expected",
            self.responsivity**2
            * Planck
            * (speed_of_light / self.WAVELENGTH)
            * self.sampling_rate
            # * signal_power(Efields),
            * signal_power(np.mean(powers)),
        )

        return currents

    def balanced_photodiodes(
        self, A_r: NDArray[np.float64], phi_r: NDArray[np.float64], lo_phase: float
    ) -> NDArray[np.float64]:
        power_diffs = A_r * self.lo_amplitude * np.cos(phi_r - lo_phase)
        photons = self.power_to_photons(power_diffs)
        print(photons, np.min(photons), np.max(photons))

        # FIXME switch to a normal distribution and stop converting to and from
        # photons.
        noisy_photons = np.random.poisson(photons)
        currents = self.responsivity * self.photons_to_power(noisy_photons)

        # print(f"{photons[:8]=}")
        # print(f"{noisy_photons[:8]=}")
        # Calculate SNR.
        signal = signal_power(photons)
        noise = signal_power(photons - noisy_photons)
        print(
            np.mean(photons),
            "photons, SNR",
            10 * np.log10(signal / noise),
            "dB",
            signal / noise,
            "lin",
        )

        return currents

    def __call__(self, Efields: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Set received signal power to that specified.
        # TODO remove this once we have proper amplitudes upstream.
        Efields *= np.sqrt(self.rx_power / signal_power(Efields))
        assert np.isclose(signal_power(Efields), self.rx_power)

        A_r = np.abs(Efields)
        phi_r: NDArray[np.float64] = np.angle(Efields)  # type: ignore

        # i1 = self.photodiode_current(A_r, phi_r, 0)
        # i2 = self.photodiode_current(A_r, phi_r, np.pi)
        # i3 = self.photodiode_current(A_r, phi_r, np.pi / 2)
        # i4 = self.photodiode_current(A_r, phi_r, np.pi * 3 / 2)

        # Compute balanced photodetectors and pack into complex numbers.
        i = np.empty_like(Efields, np.cdouble)
        # i.real = i1 - i2  # In-phase.
        # i.imag = i3 - i4  # Quadrature.
        i.real = self.balanced_photodiodes(A_r, phi_r, 0)
        i.imag = self.balanced_photodiodes(A_r, phi_r, np.pi / 2)

        expected = self.responsivity * Efields * self.lo_amplitude

        # Calculate SNR.
        signal = signal_power(expected)
        noise = np.var(i - expected)
        print(
            "FINAL SNR",
            10 * np.log10(signal / noise),
            "dB",
            signal / noise,
            "lin",
        )
        # print("expected", expected[:8])
        # print("actual", i[:8])

        # j = np.empty_like(Efields, np.cdouble)
        # j.real = A_r * self.lo_amplitude * self.responsivity * np.cos(phi_r)
        # j.imag = A_r * self.lo_amplitude * self.responsivity * np.sin(phi_r)
        # print("ideal", j[:8])

        # assert np.allclose(i, j)
        # assert np.allclose(i, expected)

        return i

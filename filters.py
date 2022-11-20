from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.constants import speed_of_light
from scipy.linalg import toeplitz
from scipy.signal import decimate
from scipy.special import erf

from utils import Component, overlap_save, signal_energy


class PulseFilter(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    # A span of 32 is quite long for high values of beta (approaching 1),
    # but it's way too short for smaller betas. 128 would be a more
    # appropriate value for betas approaching 0.
    SPAN: int = 32
    BETA: float = 0.24

    def __init__(
        self,
        samples_per_symbol: int,
        *,
        up: Optional[int] = None,
        down: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Need at least 2 samples per symbol to satisfy the Nyquist–Shannon
        # sampling theorem (raised cosine filter has bandwidth between 1/2T and
        # 1/T depending on the value of beta). This is the one-sided baseband
        # bandwidth of the pulse; even though the negative frequencies contain
        # useful information (as the signal is complex) the Nyquist frequency is
        # the same.
        assert samples_per_symbol > 1
        self.samples_per_symbol = samples_per_symbol

        # It doesn't make sense to set both.
        if up is not None:
            assert up > 1
            assert down is None
        if down is not None:
            assert down > 1
            assert up is None

        self.up = up
        self.down = down

    @cached_property
    def impulse_response(self) -> NDArray[np.float64]:
        rrc = root_raised_cosine(self.BETA, self.samples_per_symbol, self.SPAN)
        assert np.argmax(rrc) == rrc.size // 2
        assert np.isclose(signal_energy(rrc), 1)

        return rrc

    def upsample(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert self.up is not None
        assert symbols.size > 0

        # This is quite fast actually (https://stackoverflow.com/a/73994667).
        upsampled = np.zeros(symbols.size * self.up - (self.up - 1), dtype=np.cdouble)
        upsampled[:: self.up] = symbols

        return upsampled

    def subsample(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert self.down is not None
        assert symbols.size > 0

        # Don't bother copying to a new array.
        return symbols[:: self.down]

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert symbols.ndim == 1

        # To avoid aliasing, filtering should be done after upsamlping but
        # before downsampling.
        if self.up:
            symbols = self.upsample(symbols)

        # Filter the data with the impulse response of the filter.
        filtered = overlap_save(self.impulse_response, symbols, full=True)

        if self.down:
            # Should subsample after filtering to avoid aliasing.
            filtered = self.subsample(filtered)
            new_sps = self.samples_per_symbol // self.down

            # Remove RRC filter edge effects.
            return filtered[self.SPAN * new_sps : -(self.SPAN - 1) * new_sps]

        # Transmit the symbols as is.
        return filtered


def root_raised_cosine(
    beta: float, samples_per_symbol: int, span: int
) -> NDArray[np.float64]:
    assert 0 < beta < 1
    assert samples_per_symbol > 0
    assert span % 2 == 0

    # Normalize by samples_per_symbol to get time in terms of t/T
    # (T = samples_per_symbol)
    t = np.linspace(-span // 2, span // 2, samples_per_symbol * span, endpoint=False)

    assert t.size % 2 == 0
    assert t.size == samples_per_symbol * span
    assert t[samples_per_symbol * span // 2] == 0

    cos_term = np.cos((1 + beta) * np.pi * t)

    # numpy implements the normalized sinc function, so we need to divide by π
    # to obtain the unnormalized sinc(x) = sin(x)/x.
    sinc_term = np.sinc((1 - beta) * t)
    sinc_term *= (1 - beta) * np.pi / (4 * beta)

    denominator = 1 - (4 * beta * t) ** 2

    p = (cos_term + sinc_term) / denominator
    p *= 4 * beta / (np.pi * np.sqrt(samples_per_symbol))

    # FIXME have to compute the limits when |t/T| = 1/4β.
    assert np.all(np.isfinite(p))

    # Normalize energy. The equation we use does result in a unit energy signal,
    # but only if the span is infinite. Since we truncate the filter, we need to
    # re-normalize the remaining terms.
    p /= np.sqrt(signal_energy(p))

    return p


class CDBase(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    # D = 17 ps/nm/km at λ = 1550 nm according to Digital Coherent Optical
    # Systems.
    GROUP_VELOCITY_DISPERSION = 17 * 1e-12 / (1e-9 * 1e3)

    def __init__(self, length: float, sampling_rate: float) -> None:
        super().__init__()

        assert length > 0
        self.length = length

        assert sampling_rate > 0
        self.sampling_interval = 1 / sampling_rate

    @cached_property
    def K(self) -> float:
        return (
            self.GROUP_VELOCITY_DISPERSION
            * self.WAVELENGTH**2
            * self.length
            / (4 * np.pi * speed_of_light * self.sampling_interval**2)
        )


class ChromaticDispersion(CDBase):
    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert symbols.ndim == 1

        # This is the baseband representation of the signal, which has the same
        # bandwidth as the upconverted PAM signal. It's already centered around
        # 0, so there's no need to subtract the carrier frequency from its
        # spectrum.
        Df = np.fft.fftfreq(symbols.size, self.sampling_interval)

        arg = self.K * (2 * np.pi * self.sampling_interval) ** 2
        cd = np.exp(-1j * arg * Df**2)

        # FIXME this is circular convolution.
        return np.fft.ifft(np.fft.fft(symbols) * cd)


class CDCompensator(CDBase):
    """Compensate Chromatic Dispersion, using the FIR filter derived in
    A. Eghbali, H. Johansson, O. Gustafsson and S. J. Savory, "Optimal
    Least-Squares FIR Digital Filters for Compensation of Chromatic Dispersion
    in Digital Coherent Optical Receivers," in Journal of Lightwave Technology,
    vol. 32, no. 8, pp. 1449-1456, April15, 2014, doi: 10.1109/JLT.2014.2307916.
    """

    def __init__(
        self,
        length: float,
        sampling_rate: float,
        samples_per_symbol: int,
        fir_length: int,
    ) -> None:
        super().__init__(length, sampling_rate)

        assert samples_per_symbol > 0
        self.samples_per_symbol = samples_per_symbol

        # Filter length N_c is assumed to be odd. Since the bounds are from
        # -(N_c - 1)/2 to (N_c - 1)/2 then it should be at least 3.
        assert fir_length >= 3
        assert fir_length % 2 == 1
        self.fir_length = fir_length

    @cached_property
    def D(self) -> NDArray[np.cdouble]:
        half_length = (self.fir_length - 1) // 2
        n = np.arange(-half_length, half_length + 1)
        assert n.size == self.fir_length

        K = self.K
        pi_3_4 = np.pi * 3 / 4

        erf_arg = np.exp(1j * pi_3_4) / (2 * np.sqrt(K))
        erf_small = erf(erf_arg * (2 * K * np.pi - n))
        erf_large = erf(erf_arg * (2 * K * np.pi + n))

        D = erf_small + erf_large
        D *= np.exp(-1j * (n**2 / (4 * K) + pi_3_4)) / (4 * np.sqrt(np.pi * K))

        return D

    @cached_property
    def omega(self) -> float:
        return np.pi * (1 + PulseFilter.BETA) / self.samples_per_symbol

    @cached_property
    def Q(self) -> NDArray[np.float64]:
        # Q is a Hermitian Toeplitz matrix, so we only need to compute its first
        # column.
        n = np.arange(self.fir_length)

        first_column = np.empty_like(n)
        first_column[0] = self.omega / np.pi  # n = 0 is a special case.
        first_column[1:] = np.sin(n[1:] * self.omega) / (n[1:] * np.pi)

        # FIXME epsilon term.
        return toeplitz(first_column) + np.eye(self.fir_length)

    @cached_property
    def h(self) -> NDArray[np.cdouble]:
        return np.linalg.solve(self.Q, self.D)

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        return overlap_save(self.h, data, True)[
            self.fir_length // 2 : -self.fir_length // 2 + 1
        ]


class Decimate(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __init__(self, factor: int) -> None:
        super().__init__()

        assert factor > 1
        self.factor = factor

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # To prevent aliasing, a low-pass filter needs to be applied before
        # downsampling. In the real implementation, this could be done with an
        # analogue filter followed by sampling at the desired rate. Here, we
        # could do it by designing a low-pass FIR filter using the window method
        # with scipy.signal.firwin(), filtering the data ourselves, and finally
        # subsampling the signal---but for now it's easier to let
        # scipy.signal.decimate() do everything for us. By default, it runs an
        # IIR filter both forwards and backwards to ensure zero phase change.
        # TODO implement a more computationally efficient FIR method (should
        # only calculate the samples that we'll keep).
        # XXX the astype() call just makes the type checker happy; it's not
        # supposed to do anything.
        return decimate(symbols, self.factor).astype(
            np.cdouble, casting="no", copy=False
        )

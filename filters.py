from functools import cache, cached_property
from math import ceil, floor
from typing import Optional, Type

import numpy as np
from numpy.typing import NDArray
from scipy.constants import speed_of_light
from scipy.linalg import toeplitz
from scipy.signal import decimate
from scipy.special import erf

from modulation import Demodulator, Modulator
from utils import (
    Component,
    Signal,
    for_each_polarization,
    has_one_polarization,
    has_two_polarizations,
    has_up_to_two_polarizations,
    is_power_of_2,
    normalize_energy,
    normalize_power,
    overlap_save,
    row_size,
    signal_energy,
)


class PulseFilter(Component):
    # A span of 32 is quite long for high values of beta (approaching 1),
    # but it's way too short for smaller betas. 128 would be a more
    # appropriate value for betas approaching 0.
    # FIXME needs to be odd to ensure that 2^n - SPAN - 1 is even. If it's odd,
    # then we can't use Alamouti coding.
    SPAN: int = 70
    BETA: float = 0.021

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
        # XXX samples per symbol when filtering is performed.
        assert samples_per_symbol > 1
        self.samples_per_symbol = samples_per_symbol

        # It doesn't make sense to set both.
        if up is not None:
            assert up > 1
            assert down is None
        if down is not None:
            assert down > 0
            assert up is None

        self.up = up
        self.down = down

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        samples_per_symbol = self.samples_per_symbol

        if self.up:
            samples_per_symbol //= self.up

        return Signal.SYMBOLS, np.cdouble, samples_per_symbol

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        samples_per_symbol = self.samples_per_symbol

        if self.down:
            samples_per_symbol //= self.down

        return Signal.SYMBOLS, np.cdouble, samples_per_symbol

    @cached_property
    def impulse_response(self) -> NDArray[np.float64]:
        rrc = root_raised_cosine(self.BETA, self.samples_per_symbol, self.SPAN)
        assert np.argmax(rrc) == rrc.size // 2
        assert np.isclose(signal_energy(rrc), 1)

        # FIXME 2 samples per symbol but low pass filter to match the bandwidth.

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

    @for_each_polarization
    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(symbols)

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

    @classmethod
    def symbols_for_total_length(cls, total_length: int) -> int:
        return total_length - cls.SPAN + 1


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
    return normalize_energy(p)


class CDBase(Component):
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
    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.OPTICAL, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.OPTICAL, np.cdouble, None

    @staticmethod
    @cache
    def cd_spectrum(
        size: int, sampling_interval: float, K: float
    ) -> NDArray[np.cdouble]:
        # XXX this is a static method because of @cache, see Pylint W1518
        # This is the baseband representation of the signal, which has the same
        # bandwidth as the upconverted PAM signal. It's already centered around
        # 0, so there's no need to subtract the carrier frequency from its
        # spectrum.
        Df = np.fft.fftfreq(size, sampling_interval)

        arg = K * (2 * np.pi * sampling_interval) ** 2

        return np.exp(1j * arg * Df**2)

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # fft and ifft work just fine with 2D arrays. They operate on the last
        # axis by default (axis=1, i.e. along each row).
        assert has_up_to_two_polarizations(symbols)
        assert is_power_of_2(row_size(symbols))

        cd = self.cd_spectrum(row_size(symbols), self.sampling_interval, self.K)

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
        pulse_filter_beta: float = PulseFilter.BETA,
    ) -> None:
        super().__init__(length, sampling_rate)

        assert samples_per_symbol > 0
        self.samples_per_symbol = samples_per_symbol

        # Filter length N_c is assumed to be odd. Since the bounds are from
        # -(N_c - 1)/2 to (N_c - 1)/2 then it should be at least 3.
        assert fir_length >= 3
        assert fir_length % 2 == 1
        self.fir_length = fir_length

        assert 0 <= pulse_filter_beta <= 1
        self.beta = pulse_filter_beta

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, self.samples_per_symbol

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, self.samples_per_symbol

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
        return np.pi * (1 + self.beta) / self.samples_per_symbol

    @cached_property
    def Q(self) -> NDArray[np.float64]:
        # Q is a Hermitian Toeplitz matrix, so we only need to compute its first
        # column.
        n = np.arange(self.fir_length)

        first_column = np.empty_like(n, dtype=np.float64)
        first_column[0] = self.omega / np.pi  # n = 0 is a special case.
        first_column[1:] = np.sin(n[1:] * self.omega) / (n[1:] * np.pi)

        return toeplitz(first_column)

    @cached_property
    def h(self) -> NDArray[np.cdouble]:
        # Matrix becomes singular *very* quickly.
        # QQ = self.Q + 1e-14 * np.eye(self.fir_length)
        # print(QQ.shape, np.linalg.matrix_rank(QQ))
        # print(np.linalg.det(QQ))
        # print(np.linalg.cond(QQ))

        # FIXME Paper uses ε = 1e-14, although there is some flexibility.
        # return np.linalg.solve(self.Q + np.eye(self.fir_length), self.D)
        return np.conj(np.linalg.lstsq(self.Q, self.D, rcond=0.9)[0])

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(data)

        return overlap_save(self.h, data, True)[
            self.fir_length // 2 : -self.fir_length // 2 + 1
        ]


class Decimate(Component):
    def __init__(self, factor: int) -> None:
        super().__init__()

        assert factor > 1
        self.factor = factor

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # decimate operates over the last axis by default, i.e. along each row.
        assert has_up_to_two_polarizations(symbols)

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


class AdaptiveEqualizer(Component):
    cma_to_rde_threshold = 256  # TODO find a good value.

    def __init__(self, taps: int, mu: float) -> None:
        super().__init__()

        assert taps > 0
        self.taps = taps

        # FIXME 10**-2 is big, try 10**-4 to 10**-3
        # Try correcting for chromatic dispersion (mismatch)
        assert mu > 0
        self.mu = mu

        # Filter coefficients with single spike initialization.
        self.w = np.zeros(self.taps, dtype=np.cdouble)
        self.lag = floor(self.taps / 2) + 1
        self.w[self.lag - 1] = 1

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 2

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 1

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(symbols)

        normalized = normalize_power(symbols)

        # TODO train on the first e.g. 1000 symbols and then stop updating the
        # filter.

        R_cma = 1.32  # TODO explanation.
        R_rde = np.asarray((1 / np.sqrt(5), 1, 3 / np.sqrt(5)))

        # Wrap input array.
        data = np.concatenate(
            (normalized[-self.lag + 1 :], normalized, normalized[: self.lag])
        )
        assert data.size == normalized.size + self.w.size

        # Output array. Downsample 2:1. TODO add parameter for this.
        y = np.empty(ceil(normalized.size / 2), dtype=np.cdouble)

        for i in range(y.size):
            x = data[2 * i : 2 * i + self.w.size]

            y[i] = self.w.conj() @ x

            if i < self.cma_to_rde_threshold:
                # CMA update step.
                self.w += self.mu * x * (R_cma - np.abs(y[i]) ** 2) * np.conj(y[i])
            else:
                # Get radius closest to the compensated symbol.
                r = R_rde[np.argmin(np.abs(R_rde - np.abs(y[i])))]

                # RDE update step.
                self.w += self.mu * x * (r**2 - np.abs(y[i]) ** 2) * np.conj(y[i])

        return y


class AdaptiveEqualizer2P(Component):
    cma_to_rde_threshold = 256  # TODO find a good value.

    def __init__(self, taps: int, mu: float) -> None:
        super().__init__()

        assert taps > 0
        self.taps = taps

        # FIXME 10**-2 is big, try 10**-4 to 10**-3
        # Try correcting for chromatic dispersion (mismatch)
        assert mu > 0
        self.mu = mu

        # Filter coefficients.
        self.w1V = np.zeros(self.taps, dtype=np.cdouble)
        self.w1H = np.zeros(self.taps, dtype=np.cdouble)
        self.w2V = np.zeros(self.taps, dtype=np.cdouble)
        self.w2H = np.zeros(self.taps, dtype=np.cdouble)

        # Single spike initialization.
        self.lag = floor(self.taps / 2) + 1
        self.w1V[self.lag - 1] = 1

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 2

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 1

    def __call__(
        self, symbols: NDArray[np.cdouble]
    ) -> tuple[NDArray[np.cdouble], NDArray[np.cdouble]]:
        assert has_two_polarizations(symbols)

        normalized = normalize_power(symbols)

        # TODO train on the first e.g. 1000 symbols and then stop updating the
        # filter.

        R_cma = 1.32  # TODO explanation.
        R_rde = np.asarray((1 / np.sqrt(5), 1, 3 / np.sqrt(5)))

        # Wrap input array.
        dataV = np.concatenate(
            (
                normalized[0, -self.lag + 1 :],
                normalized[0, :],
                normalized[0, : self.lag],
            )
        )
        dataH = np.concatenate(
            (
                normalized[1, -self.lag + 1 :],
                normalized[1, :],
                normalized[1, : self.lag],
            )
        )
        assert dataV.size == row_size(normalized) + self.w1V.size
        assert dataH.size == row_size(normalized) + self.w1V.size

        # Output array. Downsample 2:1. TODO add parameter for this.
        y1 = np.empty(ceil(row_size(normalized) / 2), dtype=np.cdouble)
        y2 = np.empty(ceil(row_size(normalized) / 2), dtype=np.cdouble)

        for i in range(y1.size):
            xV = dataV[2 * i : 2 * i + self.w1V.size]
            xH = dataH[2 * i : 2 * i + self.w1H.size]

            y1[i] = self.w1V.conj() @ xV + self.w1H.conj() @ xH
            y2[i] = self.w2V.conj() @ xV + self.w2H.conj() @ xH

            if i < self.cma_to_rde_threshold:
                # CMA update step.
                self.w1V += self.mu * xV * (R_cma - np.abs(y1[i]) ** 2) * np.conj(y1[i])
                self.w1H += self.mu * xH * (R_cma - np.abs(y1[i]) ** 2) * np.conj(y1[i])
                self.w2V += self.mu * xV * (R_cma - np.abs(y2[i]) ** 2) * np.conj(y2[i])
                self.w2H += self.mu * xH * (R_cma - np.abs(y2[i]) ** 2) * np.conj(y2[i])
            else:
                raise NotImplementedError()
                # Get radius closest to the compensated symbol.
                # r = R_rde[np.argmin(np.abs(R_rde - np.abs(y[i])))]

                # RDE update step.
                # self.w1V += self.mu * x * (r**2 - np.abs(y[i]) ** 2) * np.conj(y[i])

            # TODO variable.
            if i == 5_000:
                # Reinitialize filter coefficients.
                self.w2H = np.conj(self.w1V[::-1])
                self.w2V = -np.conj(self.w1H[::-1])

        # FIXME should vstack.
        return y1, y2


class AdaptiveEqualizerAlamouti(Component):
    def __init__(
        self,
        taps: int,
        mu: float,
        mu_p: float,
        modulator: Modulator,
        demodulator: Demodulator,
        training_symbols: NDArray[np.cdouble],
    ) -> None:
        super().__init__()

        assert taps > 0
        self.taps = taps

        assert mu > 0
        self.mu = mu

        assert mu_p > 0
        self.mu_p = mu_p

        self.modulator = modulator
        self.demodulator = demodulator

        self.training_symbols = training_symbols

        # Filter coefficients.
        self.w11 = np.zeros(self.taps, dtype=np.cdouble)
        self.w12 = np.zeros(self.taps, dtype=np.cdouble)
        self.w21 = np.zeros(self.taps, dtype=np.cdouble)
        self.w22 = np.zeros(self.taps, dtype=np.cdouble)

        # Single-tap phase estimator.
        self.p = 1 + 0j
        self.p_1 = 1 + 0j
        self.p_2 = 1 + 0j

        # Single spike initialization.
        self.lag = floor(self.taps / 2) + 1
        self.w11[self.lag - 1] = 1
        self.w22[self.lag - 1] = -1

        self.e_o_log = []
        self.e_e_log = []
        self.p_log = []
        self.w11_log = []
        self.w12_log = []
        self.w21_log = []
        self.w22_log = []

        self.first = True

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 2

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 1

    @staticmethod
    def serial_to_parallel(
        symbols: NDArray[np.cdouble],
    ) -> tuple[NDArray[np.cdouble], NDArray[np.cdouble]]:
        assert has_one_polarization(symbols)
        assert symbols.size % 2 == 0

        # 1:2 Serial to Parallel conversion.
        normalized = symbols.reshape(-1, 2)

        # XXX odd comes first, as we use 0-based indexing.
        symbols_odd = normalized[:, 0].ravel()
        symbols_even = normalized[:, 1].ravel()

        return symbols_odd, symbols_even

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(symbols)
        assert symbols.size % 4 == 0

        # FIXME don't normalize.
        normalized = normalize_power(symbols)
        symbols_odd, symbols_even = self.serial_to_parallel(normalized)

        # Wrap input array.
        extended_odd = np.pad(symbols_odd, (self.lag - 1, self.lag), mode="wrap")
        extended_even = np.pad(symbols_even, (self.lag - 1, self.lag), mode="wrap")

        assert extended_odd.size == symbols_odd.size + self.w11.size
        assert extended_even.size == symbols_even.size + self.w22.size

        # Output array.
        y = np.empty(normalized.size, dtype=np.cdouble)

        for i in range(symbols_odd.size):
            u_o = extended_odd[i : i + self.w11.size]
            u_e = extended_even[i : i + self.w22.size]

            u_eC = np.conj(u_e)

            p = self.p
            pC = np.conj(p)
            pabs = np.abs(p)

            # Filter outputs.
            u_11 = self.w11.conj() @ u_o
            u_12 = self.w12.conj() @ u_eC
            u_21 = self.w21.conj() @ u_o
            u_22 = self.w22.conj() @ u_eC

            # Estimate next two symbols.
            v_o = u_11 * p + u_12 * pC
            v_e = u_21 * p + u_22 * pC

            if self.first and (2 * i + 2) <= self.training_symbols.size:
                d_o, d_e = self.training_symbols[2 * i : 2 * i + 2]
            else:
                # Need to modulate the decided bits again to recover their symbol.
                # FIXME verify scale = 1.
                decisions = self.demodulator(np.asarray((v_o, v_e)), 1)
                d_o, d_e = self.modulator(decisions)

            # Compute errors.
            e_o = d_o - v_o
            e_e = d_e - v_e

            self.e_o_log.append(e_o)
            self.e_e_log.append(e_e)

            # Update filter coefficients.
            self.w11 += self.mu * p / pabs * u_o * np.conj(e_o)
            self.w12 += self.mu * pC / pabs * u_eC * np.conj(e_o)
            self.w21 += self.mu * p / pabs * u_o * np.conj(e_e)
            self.w22 += self.mu * pC / pabs * u_eC * np.conj(e_e)

            # Update phase estimate.
            self.p_1 += self.mu_p * u_11 * np.conj(e_o)
            self.p_2 += self.mu_p * u_12 * np.conj(e_o)
            self.p = 0.5 * (self.p_1 + np.conj(self.p_2))

            self.p_log.append(self.p)
            self.w11_log.append(signal_energy(self.w11))
            self.w21_log.append(signal_energy(self.w21))
            self.w12_log.append(signal_energy(self.w12))
            self.w22_log.append(signal_energy(self.w22))

            # FIXME eventually output decisions.
            y[2 * i : 2 * i + 2] = v_o, v_e

        # FIXME we should be able to use training symbols after the first block
        # (e.g. if the block size is 64).
        self.first = False

        return y

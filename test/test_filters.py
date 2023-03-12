import numpy as np
import pytest
import scipy.io
from numpy.typing import NDArray

from channel import SSFChannel
from filters import (
    AdaptiveEqualizerAlamouti,
    CDCompensator,
    ChromaticDispersion,
    PulseFilter,
    root_raised_cosine,
)
from utils import is_even, normalize_energy, signal_energy

# TODO
# test for unit energy
# test for pulse bandwidth
# test for perfect recovery in the absence of noise
# test against the spectrum/series given for the raised cosine


def generate_random_data(length: int) -> NDArray[np.cdouble]:
    rng = np.random.default_rng()

    real = rng.integers(-2, 2, endpoint=True, size=length)
    imag = rng.integers(-2, 2, endpoint=True, size=length)

    return real + 1j * imag


def generate_random_pulses(length: int, samples_per_symbol: int) -> NDArray[np.cdouble]:
    data = generate_random_data(PulseFilter.symbols_for_total_length(length))
    return PulseFilter(samples_per_symbol, up=samples_per_symbol)(data)


class TestRootRaisedCosine:
    @staticmethod
    def test_low_beta():
        BETA = 0.11
        SPAN = 6
        SAMPLES_PER_SYMBOL = 4

        # Test against MATLAB's rcosdesign().
        EXPECTED = (
            -0.0136,
            0.0260,
            0.0570,
            0.0559,
            0.0144,
            -0.0509,
            -0.1024,
            -0.0985,
            -0.0149,
            0.1388,
            0.3188,
            0.4631,
            0.5181,
            0.4631,
            0.3188,
            0.1388,
            -0.0149,
            -0.0985,
            -0.1024,
            -0.0509,
            0.0144,
            0.0559,
            0.0570,
            0.0260,
        )

        result = root_raised_cosine(BETA, SAMPLES_PER_SYMBOL, SPAN)

        assert np.all(np.isfinite(result))
        assert result.ndim == 1
        assert result.size == SAMPLES_PER_SYMBOL * SPAN
        assert np.allclose(result, EXPECTED, atol=1e-4)

    @staticmethod
    def test_high_beta():
        BETA = 0.99
        SPAN = 6
        SAMPLES_PER_SYMBOL = 4

        # Test against MATLAB's rcosdesign().
        EXPECTED = (
            -0.0045,
            0.0004,
            0.0064,
            -0.0006,
            -0.0103,
            0.0006,
            0.0182,
            -0.0013,
            -0.0432,
            0.0013,
            0.2141,
            0.5000,
            0.6353,
            0.5000,
            0.2141,
            0.0013,
            -0.0432,
            -0.0013,
            0.0182,
            0.0006,
            -0.0103,
            -0.0006,
            0.0064,
            0.0004,
        )

        result = root_raised_cosine(BETA, SAMPLES_PER_SYMBOL, SPAN)

        assert np.all(np.isfinite(result))
        assert np.all(np.isreal(result))
        assert result.ndim == 1
        assert result.size == SAMPLES_PER_SYMBOL * SPAN
        assert np.allclose(result, EXPECTED, atol=1e-4)

    @staticmethod
    def test_very_low_beta() -> None:
        BETA = 0.0101  # 0.01 actually.
        SPAN = 508
        SAMPLES_PER_SYMBOL = 2

        # Test against the filter used in some of the experiments. MATLAB
        # generates completly symmetric filters with an odd number of samples,
        # so drop the last one to make its size even.
        expected = normalize_energy(
            scipy.io.loadmat("test/matlab_pf_0.01.mat")["fir"].ravel()[:-1]
        )

        result = root_raised_cosine(BETA, SAMPLES_PER_SYMBOL, SPAN)

        assert np.all(np.isfinite(result))
        assert np.all(np.isreal(result))
        assert result.ndim == 1
        assert result.size == SAMPLES_PER_SYMBOL * SPAN
        assert np.allclose(result, expected, atol=1e-4)

    @staticmethod
    @pytest.mark.parametrize("beta", (-0.1, 0, 1, 1.1))
    def test_invalid_beta(beta: float):
        with pytest.raises(Exception):
            root_raised_cosine(beta, 4, 4)

    @staticmethod
    @pytest.mark.parametrize("span", (-1, 0, 1, 3, 9))
    def test_invalid_span(span: int):
        with pytest.raises(Exception):
            root_raised_cosine(0.1, 4, span)

    @staticmethod
    @pytest.mark.parametrize("samples_per_symbol", (-1, 0))
    def test_invalid_samples_per_symbol(samples_per_symbol: int):
        with pytest.raises(Exception):
            root_raised_cosine(0.1, samples_per_symbol, 4)

    @staticmethod
    @pytest.mark.parametrize("beta", (0.01, 0.22, 0.51, 0.78, 0.99))
    @pytest.mark.parametrize("samples_per_symbol", (2, 4, 16))
    @pytest.mark.parametrize("span", (2, 4, 16))
    def test_energy_and_peak(beta: float, samples_per_symbol: int, span: int):
        result = root_raised_cosine(beta, samples_per_symbol, span)
        assert np.all(np.isreal(result))

        # Unit energy.
        assert np.isclose(signal_energy(result), 1, rtol=1e-3)

        # Peak should be in the middle (t = 0).
        assert np.argmax(result) == result.size // 2


class TestPulseFilter:
    @staticmethod
    @pytest.mark.parametrize("samples_per_symbol", (2, 4, 8, 16))
    def test_round_trip(samples_per_symbol: int):
        data = generate_random_data(2**10)

        up = PulseFilter(samples_per_symbol, up=samples_per_symbol)(data)

        assert up.ndim == 1
        assert up.dtype == np.cdouble
        # PulseFilter() keeps boundary effects when upsampling.
        assert up.size == (data.size + PulseFilter.SPAN - 1) * samples_per_symbol

        # Data is first filtered at its original rate and then it's subsampled
        # to 1 SpS.
        down = PulseFilter(samples_per_symbol, down=samples_per_symbol)(up)

        assert down.ndim == 1
        assert down.dtype == np.cdouble
        assert down.size == data.size

        # Energy need not be preserved.
        assert np.allclose(normalize_energy(down), normalize_energy(data), atol=0.01)

    @staticmethod
    @pytest.mark.parametrize("target_size", (2**7, 2**8, 2**9))
    def test_symbols_for_total_length(target_size: int) -> None:
        SAMPLES_PER_SYMBOL = 4

        pf = PulseFilter(SAMPLES_PER_SYMBOL, up=SAMPLES_PER_SYMBOL)
        length = pf.symbols_for_total_length(target_size)

        assert length > 0

        data = np.ones(length)
        up = pf(data)

        assert up.ndim == 1
        assert up.size == target_size * SAMPLES_PER_SYMBOL


class TestChromaticDispersion:
    SAMPLES_PER_SYMBOL = 4
    cd = ChromaticDispersion(10e3, 50e9 * SAMPLES_PER_SYMBOL)
    compensator = CDCompensator(10e3, 50e9 * SAMPLES_PER_SYMBOL, SAMPLES_PER_SYMBOL, 63)

    def test_spectrum(self):
        up = generate_random_pulses(2**10, self.SAMPLES_PER_SYMBOL)

        dispersed = self.cd(up)

        assert np.all(np.isfinite(dispersed))
        assert dispersed.size == up.size

        data_fft = np.fft.fft(up)
        result_fft = np.fft.fft(dispersed)

        assert np.all(np.isfinite(data_fft))
        assert np.all(np.isfinite(result_fft))

        # Only the phase of each component should had changed.
        assert np.allclose(np.abs(result_fft), np.abs(data_fft))
        assert np.any(np.angle(result_fft) != np.angle(data_fft))

    def test_compensator(self):
        up = generate_random_pulses(2**16, self.SAMPLES_PER_SYMBOL)

        dispersed = self.cd(up)

        assert np.all(np.isfinite(dispersed))
        assert dispersed.size == up.size

        filtered = self.compensator(dispersed)

        assert np.all(np.isfinite(filtered))
        assert filtered.size == dispersed.size

        # We haven't recovered the transmitted data perfectly, but hopefully
        # it's good enough.
        assert np.corrcoef(up, filtered)[1, 0] > 0.95
        assert np.isclose(signal_energy(up), signal_energy(filtered), rtol=0.01)

    def test_compensator_q(self):
        n, m = np.indices((self.compensator.fir_length, self.compensator.fir_length))
        i = m - n

        with np.errstate(invalid="ignore"):
            q_expected = (
                np.exp(1j * i * self.compensator.omega)
                - np.exp(-1j * i * self.compensator.omega)
            ) / (2j * np.pi * i)

        np.fill_diagonal(q_expected, self.compensator.omega / np.pi)

        assert np.allclose(q_expected, self.compensator.Q)


class TestSSFChannel:
    FIBRE_LENGTH = 25_000
    channel = SSFChannel(FIBRE_LENGTH, 10**11)

    @pytest.mark.parametrize("attenuation", (0.0, 1e-4, 1e-3, 2e-3))
    def test_attenuation(self, attenuation: float) -> None:
        tx = generate_random_pulses(2**10, 2)

        self.channel.ATTENUATION = attenuation
        rx = self.channel(tx)

        assert np.all(np.isfinite(rx))
        assert rx.size == tx.size

        # Energy should follow Beer's law.
        expected_energy = signal_energy(tx) * np.exp(-attenuation * self.FIBRE_LENGTH)
        assert np.isclose(expected_energy, signal_energy(rx))


class TestAdaptiveEqualizerAlamouti:
    @staticmethod
    def test_serial_to_parallel() -> None:
        symbols = np.arange(1, 9, dtype=np.cdouble)

        assert is_even(symbols.size)

        odd, even = AdaptiveEqualizerAlamouti.serial_to_parallel(symbols)

        assert odd.dtype == even.dtype == symbols.dtype
        assert odd.size == even.size == symbols.size // 2

        assert np.all(odd.real % 2 == 1)
        assert np.all(even.real % 2 == 0)

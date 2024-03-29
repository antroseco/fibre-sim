import numpy as np
import pytest
import scipy.signal
from numpy.typing import NDArray

from channel import Splitter, SSFChannel
from filters import CDCompensator, PulseFilter
from frequency_recovery import FrequencyRecoveryFFT, FrequencyRecoveryLiChen
from laser import NoisyLaser
from modulation import IQModulator, Modulator16QAM
from receiver import Digital90degHybrid, HeterodyneFrontEnd

CHANNEL_SPS = 16
RECEIVER_SPS = 4  # Ensure no aliasing.
SYMBOL_RATE = 50e9

IIRFILT = scipy.signal.butter(
    4, 65e9, analog=False, output="sos", fs=SYMBOL_RATE * CHANNEL_SPS
)


def generate_symbols(length: int, f_offset_GHz: float) -> NDArray[np.cdouble]:
    # Generate and modulate data.
    num_bits = (
        PulseFilter.symbols_for_total_length(length) * Modulator16QAM.bits_per_symbol
    )
    data = np.random.randint(0, 2, num_bits, np.bool_)
    tx_mod = Modulator16QAM()(data)

    tx_pf = PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS)(tx_mod)

    # Modulate laser.
    laser = NoisyLaser(10, SYMBOL_RATE * CHANNEL_SPS)
    tx_txd = IQModulator(laser)(tx_pf)

    # Simulate channel.
    ch_1 = SSFChannel(24_000, SYMBOL_RATE * CHANNEL_SPS)(tx_txd)
    ch_s = Splitter(32)(ch_1)
    ch_2 = SSFChannel(1_000, SYMBOL_RATE * CHANNEL_SPS)(ch_s)

    # Heterodyne detector. 26 GHz base offset.
    rx_fe = HeterodyneFrontEnd(26 + f_offset_GHz, SYMBOL_RATE * CHANNEL_SPS)(ch_2)

    rx_filt = scipy.signal.sosfiltfilt(IIRFILT, rx_fe)[:: CHANNEL_SPS // RECEIVER_SPS]

    # Translate spectrum to the left by 26 GHz.
    rx_hybrid = Digital90degHybrid(26, SYMBOL_RATE * RECEIVER_SPS)(rx_filt)

    # Low-pass filter to 26 GHz.
    lp_filt = scipy.signal.firwin(127, 26e9, fs=SYMBOL_RATE * RECEIVER_SPS)
    rx_lp = scipy.signal.filtfilt(lp_filt, 1, rx_hybrid)

    # CD compensation must be performed after (coarse) frequency offset
    # compensation.
    rx_cdc = CDCompensator(25_000, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, 127)(rx_lp)

    # Ready for frequency recovery.
    return rx_cdc


class TestFrequencyRecoveryFFT:
    @staticmethod
    @pytest.mark.parametrize("freq_offset_GHz", np.linspace(-1.2, 1.2, 4))
    def test_estimate(freq_offset_GHz: float) -> None:
        # Use a long FFT to reduce the test's variability.
        fr = FrequencyRecoveryFFT(SYMBOL_RATE, RECEIVER_SPS, 1024, "gaussian")

        # Get an average---we are concerned about steady-state performance.
        estimates = []
        for _ in range(8):
            symbols = generate_symbols(4096, freq_offset_GHz)

            fr(symbols)

            assert fr.freq_estimate is not None
            estimates.append(fr.freq_estimate)

        # Large tolerance as phase recovery can handle the difference.
        assert np.isclose(np.mean(estimates), freq_offset_GHz * 1e9, atol=20e6)


class TestFrequencyRecoveryLiChen:
    @staticmethod
    @pytest.mark.parametrize("freq_offset_GHz", np.linspace(-0.7, 0.7, 4))
    def test_estimate(freq_offset_GHz: float) -> None:
        # Use a long sample size to reduce the test's variability.
        fr = FrequencyRecoveryLiChen(SYMBOL_RATE, RECEIVER_SPS, 832)

        # Get an average---we are concerned about steady-state performance.
        estimates = []
        for _ in range(32):
            symbols = generate_symbols(2048, freq_offset_GHz)

            fr(symbols)

            assert fr.freq_estimate is not None
            estimates.append(fr.freq_estimate)

        # Even larger tolerance; now 300 MHz. This method appears to have
        # significant variance.
        assert np.isclose(np.mean(estimates), freq_offset_GHz * 1e9, atol=300e6)

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from channel import Splitter, SSFChannel
from filters import CDCompensator, PulseFilter
from frequency_recovery import FrequencyRecovery
from laser import NoisyLaser
from modulation import IQModulator, ModulatorQPSK
from receiver import Digital90degHybrid, HeterodyneFrontEnd


class TestFrequencyRecovery:
    CHANNEL_SPS = 16
    RECEIVER_SPS = 2
    SYMBOL_RATE = 50 * 10**9

    def generate_symbols(self, length: int) -> NDArray[np.cdouble]:
        # Generate and modulate data.
        data = np.random.randint(0, 1, length * ModulatorQPSK.bits_per_symbol, np.bool_)
        tx_mod = ModulatorQPSK()(data)
        tx_pf = PulseFilter(self.CHANNEL_SPS, up=self.CHANNEL_SPS)(tx_mod)

        # Modulate laser.
        laser = NoisyLaser(8, self.SYMBOL_RATE * self.CHANNEL_SPS)
        tx_txd = IQModulator(laser)(tx_pf)

        # Simulate channel.
        ch_1 = SSFChannel(24_000, self.SYMBOL_RATE * self.CHANNEL_SPS)(tx_txd)
        ch_s = Splitter(64)(ch_1)
        ch_2 = SSFChannel(1_000, self.SYMBOL_RATE * self.CHANNEL_SPS)(ch_s)

        # Heterodyne detector.
        rx_fe = HeterodyneFrontEnd(25.2, self.SYMBOL_RATE * self.CHANNEL_SPS)(ch_2)

        # TODO replace this with whatever component we implement.
        # Design Butterworth IIR filter. You could implement this with analog
        # electronics if you had opamps with sufficient bandwidth.
        # XXX we filter with sosfiltfilt, so the filter order is doubled.
        iirfilt = scipy.signal.butter(
            4, 65e9, analog=False, output="sos", fs=self.CHANNEL_SPS * self.SYMBOL_RATE
        )
        rx_filt = scipy.signal.sosfiltfilt(iirfilt, rx_fe)[
            :: self.CHANNEL_SPS // self.RECEIVER_SPS
        ]

        # Translate spectrum to the left by 25.1 GHz.
        rx_hybrid = Digital90degHybrid(25.1, self.SYMBOL_RATE * self.RECEIVER_SPS)(
            rx_filt
        )

        rx_cdc = CDCompensator(
            25_000, self.SYMBOL_RATE * self.RECEIVER_SPS, self.RECEIVER_SPS, 63
        )(rx_hybrid)

        # Ready for frequency recovery.
        return rx_cdc

    def test_estimate(self) -> None:
        symbols = self.generate_symbols(PulseFilter.symbols_for_total_length(512))

        fr = FrequencyRecovery(self.RECEIVER_SPS * self.SYMBOL_RATE)
        fr.estimate(symbols)

        assert fr.freq_estimate is not None
        assert np.isclose(fr.freq_estimate, 100e6, atol=4e6)

    def test_correction(self) -> None:
        symbols = self.generate_symbols(PulseFilter.symbols_for_total_length(512))

        fr = FrequencyRecovery(self.RECEIVER_SPS * self.SYMBOL_RATE)
        corrected = fr(symbols)

        assert corrected.size == symbols.size
        assert corrected.dtype == symbols.dtype

        # Estimate new IF. Zero-padding helps increase frequency resolution.
        # Windowing increases robustness.
        window = scipy.signal.get_window("hamming", corrected.size)
        fft = np.fft.fft(corrected**4 * window, n=4096)
        freqs = np.fft.fftfreq(4096, 1 / (self.RECEIVER_SPS * self.SYMBOL_RATE))

        peak_freq = freqs[np.argmax(np.abs(fft))]

        assert np.abs(peak_freq) <= 5e6

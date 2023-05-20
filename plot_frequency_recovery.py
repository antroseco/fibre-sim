import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import signal

from channel import Splitter, SSFChannel
from filters import CDCompensator, PulseFilter
from frequency_recovery import FrequencyRecoveryFFT, FrequencyRecoveryLiChen
from laser import NoisyLaser
from modulation import IQModulator, Modulator16QAM
from receiver import Digital90degHybrid, HeterodyneFrontEnd

CHANNEL_SPS = 16
RECEIVER_SPS = 4
SYMBOL_RATE = 50e9

LENGTH = PulseFilter.symbols_for_total_length(4096)
F_OFFSETS = np.linspace(-2.5, 2.5, 32, endpoint=True)

IIRFILT = signal.butter(
    4, 65e9, analog=False, output="sos", fs=SYMBOL_RATE * CHANNEL_SPS
)

LP_FILTER_POLE = 26.0e9  # TODO vary


def do_one() -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    # Generate and modulate data.
    data = np.random.randint(0, 2, LENGTH * Modulator16QAM.bits_per_symbol, np.bool_)
    tx_mod = Modulator16QAM()(data)

    tx_pf = PulseFilter(CHANNEL_SPS, up=CHANNEL_SPS)(tx_mod)

    # Modulate laser.
    laser = NoisyLaser(10, SYMBOL_RATE * CHANNEL_SPS)
    # print(f"Laser power {10 * np.log10(signal_power(laser(100)) / 1e-3):.2f} dBm")
    tx_txd = IQModulator(laser)(tx_pf)

    # FIXME print(f"TX power {10 * np.log10(signal_power(tx_txd) / 1e-3):.2f} dBm")

    # Simulate channel.
    ch_1 = SSFChannel(24_000, SYMBOL_RATE * CHANNEL_SPS)(tx_txd)
    ch_s = Splitter(32)(ch_1)
    ch_2 = SSFChannel(1_000, SYMBOL_RATE * CHANNEL_SPS)(ch_s)

    # print(f"RX power {10 * np.log10(signal_power(ch_2) / 1e-3):.2f} dBm")

    # Per frequency offset.
    diff_384_estimates = []
    diff_832_estimates = []
    fft_256_estimates = []
    fft_512_estimates = []
    fft_1024_estimates = []

    for f_offset in F_OFFSETS:
        # Heterodyne detector (adds 26+f_offset GHz IF).
        rx_fe = HeterodyneFrontEnd(26 + f_offset, SYMBOL_RATE * CHANNEL_SPS)(ch_2)

        # Low-pass filter and downsample.
        rx_filt = signal.sosfiltfilt(IIRFILT, rx_fe)[:: CHANNEL_SPS // RECEIVER_SPS]

        # Translate spectrum to the left by 26 GHz.
        rx_hybrid = Digital90degHybrid(26, SYMBOL_RATE * RECEIVER_SPS)(rx_filt)

        # Low-pass filter to a bit over 25 GHz.
        lp_filt = signal.firwin(127, LP_FILTER_POLE, fs=SYMBOL_RATE * RECEIVER_SPS)
        rx_lp = signal.filtfilt(lp_filt, 1, rx_hybrid)

        # CD compensation must be performed after (coarse) frequency offset
        # compensation.
        rx_cdc = CDCompensator(25_000, SYMBOL_RATE * RECEIVER_SPS, RECEIVER_SPS, 127)(
            rx_lp
        )

        fr384 = FrequencyRecoveryLiChen(SYMBOL_RATE, RECEIVER_SPS, 384)
        fr384(rx_cdc)
        diff_384_estimates.append(fr384.freq_estimate)

        fr832 = FrequencyRecoveryLiChen(SYMBOL_RATE, RECEIVER_SPS, 832)
        fr832(rx_cdc)
        diff_832_estimates.append(fr832.freq_estimate)

        fft256 = FrequencyRecoveryFFT(SYMBOL_RATE, RECEIVER_SPS, 256, "gaussian")
        fft256(rx_cdc)
        fft_256_estimates.append(fft256.freq_estimate)

        fft512 = FrequencyRecoveryFFT(SYMBOL_RATE, RECEIVER_SPS, 512, "gaussian")
        fft512(rx_cdc)
        fft_512_estimates.append(fft512.freq_estimate)

        fft1024 = FrequencyRecoveryFFT(SYMBOL_RATE, RECEIVER_SPS, 1024, "gaussian")
        fft1024(rx_cdc)
        fft_1024_estimates.append(fft1024.freq_estimate)

    return (
        diff_384_estimates,
        diff_832_estimates,
        fft_256_estimates,
        fft_512_estimates,
        fft_1024_estimates,
    )


def main() -> None:
    diff_384_samples = []
    diff_832_samples = []
    fft_256_samples = []
    fft_512_samples = []
    fft_1024_samples = []

    N = 128

    for _ in range(N):
        (
            diff_384_estimates,
            diff_832_estimates,
            fft_256_estimates,
            fft_512_estimates,
            fft_1024_estimates,
        ) = do_one()

        diff_384_samples.append(diff_384_estimates)
        diff_832_samples.append(diff_832_estimates)
        fft_256_samples.append(fft_256_estimates)
        fft_512_samples.append(fft_512_estimates)
        fft_1024_samples.append(fft_1024_estimates)

    def compute_mean(samples: list[list[float]]) -> NDArray[np.float64]:
        # Transpose and sum to get a list of means by frequency.
        return np.fromiter(
            (sum(lst) / N for lst in zip(*samples)),
            dtype=np.float64,
            count=F_OFFSETS.size,
        )

    mean_diff_384 = compute_mean(diff_384_samples)
    mean_diff_832 = compute_mean(diff_832_samples)
    mean_fft_256 = compute_mean(fft_256_samples)
    mean_fft_512 = compute_mean(fft_512_samples)
    mean_fft_1024 = compute_mean(fft_1024_samples)

    def do_plot(estimates: NDArray[np.float64], marker: str, label: str) -> None:
        plt.plot(
            F_OFFSETS,
            estimates / 1e9,  # in GHz.
            marker,
            alpha=0.6,
            label=label,
        )

    plt.plot(F_OFFSETS, F_OFFSETS, alpha=0.5, lw=5, label="Target")
    do_plot(mean_diff_832, "s-", "Li and Chen (832)")
    do_plot(mean_diff_384, "s-", "Li and Chen (384)")
    do_plot(mean_fft_1024, "o-", "1024-FFT")
    do_plot(mean_fft_512, "o-", "512-FFT")
    do_plot(mean_fft_256, "o-", "256-FFT")
    plt.xlabel("True frequency offset (GHz)")
    plt.ylabel("Estimated frequency offset (GHz)")
    plt.title(
        f"Low-pass filter pole: {LP_FILTER_POLE / 1e9:.1f} GHz; "
        f"Averaged over {N} blocks"
    )
    plt.legend()
    plt.axis("square")
    plt.show()

    def compute_std(samples: list[list[float]]) -> NDArray[np.float64]:
        # Transpose and compute standard deviation by frequency.
        return np.fromiter(
            (np.std(lst, ddof=1) for lst in zip(*samples)),
            dtype=np.float64,
            count=F_OFFSETS.size,
        )

    std_diff_384 = compute_std(diff_384_samples)
    std_diff_832 = compute_std(diff_832_samples)
    std_fft_256 = compute_std(fft_256_samples)
    std_fft_512 = compute_std(fft_512_samples)
    std_fft_1024 = compute_std(fft_1024_samples)

    do_plot(std_diff_832, "s-", "Li and Chen (832)")
    do_plot(std_diff_384, "s-", "Li and Chen (384)")
    do_plot(std_fft_1024, "o-", "1024-FFT")
    do_plot(std_fft_512, "o-", "512-FFT")
    do_plot(std_fft_256, "o-", "256-FFT")
    plt.xlabel("True frequency offset (GHz)")
    plt.ylabel("Estimate standard deviation (GHz)")
    plt.title(
        f"Low-pass filter pole: {LP_FILTER_POLE / 1e9:.1f} GHz; "
        f"Standard deviation over {N} blocks"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

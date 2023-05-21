from typing import Any, TypeGuard

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import io as sio
from scipy import signal

from filters import AdaptiveEqualizerAlamouti, CDCompensator, PulseFilter
from modulation import Demodulator16QAM, Modulator16QAM
from receiver import Digital90degHybrid
from utils import normalize_power

plt.rcParams.update({"font.size": 14})


# 32k 16-QAM symbols, repeated twice.
qpsk_sync = np.tile(np.ravel(sio.loadmat("data_alamouti/QPSK_sync.mat")["s"]), 2)


data_ref = np.ravel(sio.loadmat("data_alamouti/AC_data.mat")["s_qam"]) * (
    1 / np.sqrt(10)
)

data_training = np.tile(data_ref, 4)
data_training_demod = Demodulator16QAM()(data_training)


def find_lag(in1: NDArray[np.cdouble], in2: NDArray[np.cdouble]) -> int:
    corr = np.abs(signal.correlate(in1, in2))
    lags = signal.correlation_lags(in1.size, in2.size)

    return lags[np.argmax(corr)]


def is_cdouble_array(value: Any) -> TypeGuard[NDArray[np.cdouble]]:
    return isinstance(value, np.ndarray) and value.dtype == np.cdouble


def load_sample_data(data_path: str) -> NDArray[np.cdouble]:
    # 256 GSa/s, stored as 16-bit signed integers.
    # Convert to cdouble and normalize power.
    data_256i = np.ravel(sio.loadmat(data_path)["ch1"])

    # NOTE the mean is not 0, but subtracting it did not help at all.
    data_256d = data_256i.astype(np.cdouble)

    return normalize_power(data_256d)


def run_front_end(data_256d: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
    # Peak in the spectrum is the IF. Round to 3 decimal places; we shouldn't
    # have too high a precision.
    spectrum = np.abs(np.fft.fft(data_256d))
    freqs = np.fft.fftfreq(data_256d.size, d=1 / 256e9)

    # Erase peaks outside the 26 GHz region.
    spectrum[freqs < 24e9] = 0

    if_freq_GHz = round(
        freqs[np.argmax(spectrum)] / 1e9,
        3,
    )

    if abs(if_freq_GHz - 26) > 0.5:
        print(f"WARNING: IF = {if_freq_GHz} GHz")

    # XXX Do this first before resampling to 100 GS/s. Note that this IF is not
    # negative, unlike what we found when resample first!
    downconverted = Digital90degHybrid(abs(if_freq_GHz), 256e9)(data_256d)

    # Resample to 100 GSa/s (2 SpS)
    data_100 = signal.resample_poly(downconverted, up=25, down=64)

    assert is_cdouble_array(data_100)
    return data_100


def run_static_equalization(data_100: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
    # Compensate for 25 km of fibre. TODO vary filter length.
    rx_cd = CDCompensator(25_000, 100e9, 2, 35, 0.01)(data_100)

    # Apply matched filter.
    pf = PulseFilter(2, down=1)
    pf.BETA = 0.01
    pf.SPAN = 508
    assert pf.impulse_response.size == 1016

    return pf(rx_cd)


def extract_first_frame(rx_pf: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
    # NOTE downsample to 1 SpS to match qpsk_sync.
    lag = find_lag(rx_pf[0:100_000:2], np.conj(qpsk_sync))
    assert lag >= 0

    # Multiply lag by 2 as rx_pf is at 2 SpS.
    rx_sync = rx_pf[2 * lag :]

    # Extract a single 16-QAM frame. Don't forget that it's a 2 SpS signal.
    header_size = qpsk_sync.size
    block_size = data_ref.size

    return rx_sync[2 * header_size :][: 2 * block_size]


def demodulate(rx_16qam: NDArray[np.cdouble]) -> float:
    rx_input = np.tile(rx_16qam, 4)

    # Different SpS.
    assert rx_input.size == 2 * data_training.size

    aeq = AdaptiveEqualizerAlamouti(
        49, 5e-4, 0.08, Modulator16QAM(), Demodulator16QAM(), data_training, False
    )

    rx_aeq = aeq(rx_input[0::2], rx_input[1::2])

    # Take the last 24_000 symbols; this should avoid edge effects.
    rx_demod = Demodulator16QAM()(rx_aeq[-24_000:])

    # Compute BER.
    ber = np.mean(rx_demod ^ data_training_demod[-rx_demod.size :])
    assert isinstance(ber, float)

    return ber


def slog10(num: float) -> str:
    return f"{np.log10(num):.2f}" if num != 0 else "-inf "


def process_file(data_path: str) -> float:
    data_recv = load_sample_data(data_path)

    data_100 = run_front_end(data_recv)
    rx_pf = run_static_equalization(data_100)
    rx_16qam = extract_first_frame(rx_pf)
    ber = demodulate(rx_16qam)

    print(f"{data_path} BER: {ber:.2e}, log10(BER): {slog10(ber)}")

    return ber


def main() -> None:
    # Have data from -25 dBm to -19 dBm, inclusive.
    dbms = range(-25, -18)
    bers_1 = np.fromiter(
        map(
            lambda i: process_file(f"data_bence_paper/capture_50G_run5_{i}dBm.mat"),
            dbms,
        ),
        np.float64,
    )
    bers_2 = np.fromiter(
        map(
            lambda i: process_file(f"data_bence_paper/capture_50G_run6_{i}dBm.mat"),
            dbms,
        ),
        np.float64,
    )

    fig, ax = plt.subplots()

    ax.plot(dbms, np.log10(bers_1), label="Run 1", alpha=0.6, linewidth=2, marker="o")
    ax.plot(dbms, np.log10(bers_2), label="Run 2", alpha=0.6, linewidth=2, marker="s")

    orig_lims = ax.get_xlim()
    ax.hlines(-2, *orig_lims, label="FEC limit", alpha=0.4, linewidth=4)
    ax.set_xlim(orig_lims)

    ax.set_xlabel("Received Power [dBm]")
    ax.set_ylabel(r"$\log_{10}$BER")
    ax.legend()
    ax.set_title("16-QAM, 50 GBd, 25 km")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

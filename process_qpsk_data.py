import numpy as np
import scipy.io
import scipy.signal
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from filters import AdaptiveEqualizer2P, CDCompensator, Decimate
from modulation import DemodulatorDQPSK, DemodulatorQPSK, ModulatorQPSK
from phase_recovery import DecisionDirected, ViterbiViterbi
from utils import ints_to_bits

plt.rcParams.update({"font.size": 14})


def demodulate(
    data: NDArray[np.float64], dd_phase_recovery: bool
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    assert data.shape[1] == 4

    # Extract polarizations.
    x = data[:, 1] + 1j * data[:, 0]
    y = data[:, 3] + 1j * data[:, 2]

    # Need to decimate from 100 GSamples/s to 20 GSamples/s
    # (2 SpS @ 10 GBaud)
    decimate = Decimate(5)
    x_d = decimate(x)
    y_d = decimate(y)

    # Compensate for Chromatic Dispersion (100 km transmission, so ~1600 ps/nm).
    cdcompensator = CDCompensator(100_000, 20e9, 2, 127, pulse_filter_beta=0.7)
    x_cd = cdcompensator(x_d)
    y_cd = cdcompensator(y_d)

    # CMA equalization. NOTE the adaptive equalizer DOWNSAMPLES 2 to 1.
    a_eq = AdaptiveEqualizer2P(5, 1e-3)
    a_eq.cma_to_rde_threshold = x.size * 2  # (only use CMA)
    x_eq, y_eq = a_eq(np.row_stack((x_cd, y_cd)))

    # Drop first 40_000 symbols, to ensure the adaptive equalizer has converged.
    x_eq = x_eq[40_000:]
    y_eq = y_eq[40_000:]

    # Phase recovery.
    if dd_phase_recovery:
        vv = DecisionDirected(ModulatorQPSK(), DemodulatorQPSK(), 64, 10e9, 400e3, 10)
        x_vv = ModulatorQPSK()(vv(x_eq))
        y_vv = ModulatorQPSK()(vv(y_eq))
    else:
        vv = ViterbiViterbi(ModulatorQPSK(), DemodulatorQPSK(), 64, 10e9, 400e3, 10)
        x_vv = vv(x_eq)
        y_vv = vv(y_eq)

    demod = DemodulatorDQPSK()

    # We don't have sufficient information to decode the first symbol.
    return demod(x_vv)[2:], demod(y_vv)[2:]


def find_lag(ref: NDArray[np.bool_], sig: NDArray[np.bool_]) -> int:
    corr = scipy.signal.correlate(ref.astype(np.float64), sig.astype(np.float64))
    lags = scipy.signal.correlation_lags(ref.size, sig.size)

    lag = lags[np.argmax(corr)]

    return lag


def compare_streams(
    ref: NDArray[np.bool_], sig: NDArray[np.bool_]
) -> tuple[float, float]:
    # Ensure lag + y.size < x.size.
    sig_slice = sig[:-10_000] if sig.size > ref.size - 10_000 else sig

    # Both polarizations carry the same data, but one stream is delayed by 98
    # symbols (196 bits). Unfortunately, this isn't always the case.
    lag = find_lag(ref, sig_slice)
    assert lag >= 0

    ref_slice = ref[lag : lag + sig_slice.size]
    assert ref_slice.size == sig_slice.size

    ber = np.count_nonzero(ref_slice ^ sig_slice) / ref_slice.size

    return lag, ber


def slog10(num: float) -> str:
    return f"{np.log10(num):.2f}" if num != 0 else "-inf "


def process_file(data_path: str, dd_phase_recovery: bool) -> tuple[float, float]:
    # Rotate by 45 deg to correct the weird constellation.
    data_recv = scipy.io.loadmat(data_path)["data"] * np.exp(1j * np.pi / 4)

    # Decoded DQPSK data.
    data_refd = scipy.io.loadmat("data/refdata512.mat")["refdatad"].ravel()

    # Just need to turn it into bits to match our demodulator.
    data_refd_demod = ints_to_bits(data_refd, 2)

    assert data_recv.shape == (1_000_000, 4)

    x_demod, y_demod = demodulate(data_recv, dd_phase_recovery)
    x_lag, x_ber = compare_streams(data_refd_demod, x_demod)
    y_lag, y_ber = compare_streams(data_refd_demod, y_demod)
    print(
        f"{data_path} log10(BER): {slog10(x_ber)} @ {x_lag:5}, "
        f"{slog10(y_ber)} @ {y_lag:5}"
    )

    xy_lag = np.abs(find_lag(x_demod, y_demod))
    if not (194 <= xy_lag <= 198):
        print(f"{data_path} WARNING x-y lag {xy_lag}")

    # Polarizations might flip, this keeps things consistent.
    return max(x_ber, y_ber), min(x_ber, y_ber)


def main() -> None:
    dbms = range(-50, -33)
    results_vv = map(lambda i: process_file(f"data/data_{np.abs(i)}.mat", False), dbms)
    results_dd = map(lambda i: process_file(f"data/data_{np.abs(i)}.mat", True), dbms)
    x_bers_vv, y_bers_vv = zip(*results_vv)
    x_bers_dd, y_bers_dd = zip(*results_dd)

    # TODO fit theoretical curve (2x BER of QPSK).

    fig, ax = plt.subplots()

    ax.plot(dbms, x_bers_vv, label="X pol. (VV)", alpha=0.6, linewidth=2, marker="o")
    ax.plot(dbms, y_bers_vv, label="Y pol. (VV)", alpha=0.6, linewidth=2, marker="o")
    ax.plot(dbms, x_bers_dd, label="X pol. (DD)", alpha=0.6, linewidth=2, marker="o")
    ax.plot(dbms, y_bers_dd, label="Y pol. (DD)", alpha=0.6, linewidth=2, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Received Power [dBm]")
    ax.set_ylabel("BER")
    ax.legend()
    ax.set_title("DP-DQPSK, 10 GBd, 100 km")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

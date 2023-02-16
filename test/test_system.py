from typing import Optional, Type

import numpy as np
import pytest
from numpy.typing import NDArray

from channel import AWGN
from data_stream import PseudoRandomStream
from filters import Decimate, PulseFilter
from modulation import (
    Demodulator16QAM,
    DemodulatorBPSK,
    DemodulatorQPSK,
    Modulator16QAM,
    ModulatorBPSK,
    ModulatorQPSK,
)
from system import MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, build_system
from utils import (
    Component,
    Signal,
    calculate_awgn_ber_with_bpsk,
    calculate_awgn_ser_with_qam,
    signal_energy,
)


class Counter(Component):
    def __init__(self) -> None:
        super().__init__()

        self.calls = 0
        self.count = 0

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        self.calls += 1
        self.count += data.size

        return data

    def reset(self) -> None:
        self.calls = 0
        self.count = 0


class EnergySensor(Component):
    def __init__(self) -> None:
        super().__init__()

        self.mean: float = 0
        self.count: int = 0

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Symbol energy is defined as the expected value of |x_k|^2 for all
        # constellation symbols x_k.
        energy = signal_energy(symbols)
        total = self.mean * self.count + energy

        self.count += symbols.size
        self.mean = total / self.count

        return symbols

    def reset(self) -> None:
        self.mean = 0
        self.count = 0


class TestSystem:
    def test_typecheck_system(self):
        # TODO
        pass

    def test_build_system(self):
        counter = Counter()
        data_stream = PseudoRandomStream()

        # Test simple data passthrough (modulation is required).
        system = build_system(
            data_stream, (ModulatorBPSK(), counter, DemodulatorBPSK())
        )

        bit_errors = system(MIN_CHUNK_SIZE)

        # Check that no errors have been reported.
        assert bit_errors == 0

        # Check that all data has passed through the channel.
        assert counter.calls == 1
        assert counter.count == MIN_CHUNK_SIZE

        counter.reset()

        # Test automatic fragmentation.
        bit_errors = system(2 * MAX_CHUNK_SIZE + MIN_CHUNK_SIZE - 1)

        # Check that no errors have been reported.
        assert bit_errors == 0

        # Check that the data has been split in 3. Last block is too small and
        # should be dropped automatically.
        assert counter.calls == 2
        assert counter.count == 2 * MAX_CHUNK_SIZE


class TestIntegration:
    energy_sensor = EnergySensor()

    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    def test_bpsk_over_awgn(self, eb_n0: float):
        LENGTH = 2**20

        config = (
            ModulatorBPSK(),
            self.energy_sensor,
            AWGN(eb_n0 * ModulatorBPSK.bits_per_symbol, 1),
            DemodulatorBPSK(),
        )
        system = build_system(PseudoRandomStream(), config)

        bit_errors = system(LENGTH)

        # Check with the theoretical rate.
        theoretical_ber = calculate_awgn_ber_with_bpsk(np.asarray(eb_n0))
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

        # All symbols should have unit energy.
        assert np.isclose(self.energy_sensor.mean, 1)

    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    def test_qpsk_over_awgn(self, eb_n0: float):
        LENGTH = 2**20

        config = (
            ModulatorQPSK(),
            self.energy_sensor,
            AWGN(eb_n0 * ModulatorQPSK.bits_per_symbol, 1),
            DemodulatorQPSK(),
        )
        system = build_system(PseudoRandomStream(), config)

        bit_errors = system(LENGTH)

        # Check with the theoretical rate.
        # QPSK bit error rate is equal to the BPSK bit error rate.
        theoretical_ber = calculate_awgn_ber_with_bpsk(np.asarray(eb_n0))
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

        # All symbols should have unit energy.
        assert np.isclose(self.energy_sensor.mean, 1)

    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    def test_16qam_over_awgn(self, eb_n0: float):
        LENGTH = 2**20

        config = (
            Modulator16QAM(),
            self.energy_sensor,
            AWGN(eb_n0 * Modulator16QAM.bits_per_symbol, 1),
            Demodulator16QAM(),
        )
        system = build_system(PseudoRandomStream(), config)

        bit_errors = system(LENGTH)

        # Check with the theoretical rate.
        theoretical_ser = calculate_awgn_ser_with_qam(16, np.asarray(eb_n0))
        theoretical_ber = theoretical_ser / 4
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

        # All symbols should have unit energy.
        assert np.isclose(self.energy_sensor.mean, 1, atol=1e-3)

    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    @pytest.mark.parametrize("channel_sps", [4, 8, 16, 32])
    def test_16qam_with_pulse_shaping(self, eb_n0: float, channel_sps: int):
        LENGTH = 2**18
        receiver_sps = channel_sps // 2

        config = (
            Modulator16QAM(),
            PulseFilter(channel_sps, up=channel_sps),
            Decimate(2),
            # Noise applied at the receiver.
            AWGN(eb_n0 * Modulator16QAM.bits_per_symbol, receiver_sps),
            # PulseFilter filters at receiver_sps and then subsamples to 1 SpS.
            PulseFilter(receiver_sps, down=receiver_sps),
            Demodulator16QAM(),
        )
        system = build_system(PseudoRandomStream(), config)

        bit_errors = system(LENGTH)

        # Check with the theoretical rate.
        theoretical_ser = calculate_awgn_ser_with_qam(16, np.asarray(eb_n0))
        theoretical_ber = theoretical_ser / 4
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

import numpy as np
import pytest
from channel import AWGN
from data_stream import PseudoRandomStream
from filters import PulseFilter
from modulation import (
    Demodulator16QAM,
    DemodulatorBPSK,
    DemodulatorQPSK,
    Modulator16QAM,
    ModulatorBPSK,
    ModulatorQPSK,
)
from numpy.typing import NDArray
from system import MAX_CHUNK_SIZE, build_system
from utils import (
    Component,
    calculate_awgn_ber_with_bpsk,
    calculate_awgn_ser_with_qam,
    calculate_n0,
)


class Counter(Component):
    input_type = "bits"
    output_type = "bits"

    def __init__(self) -> None:
        super().__init__()

        self.calls = 0
        self.count = 0

    def __call__(self, data: NDArray[np.bool8]) -> NDArray[np.bool8]:
        self.calls += 1
        self.count += data.size

        return data

    def reset(self) -> None:
        self.calls = 0
        self.count = 0


class EnergySensor(Component):
    input_type = "cd symbols"
    output_type = "cd symbols"

    def __init__(self) -> None:
        super().__init__()

        self.mean: float = 0
        self.count: int = 0

    def __call__(self, symbols: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        # Symbol energy is defined as the expected value of |x_k|^2 for all
        # constellation symbols x_k.
        energy = np.sum(np.abs(symbols) ** 2)
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

        # Test simple data passthrough.
        system = build_system(data_stream, (counter,))

        bit_errors = system(2)

        # Check that no errors have been reported.
        assert bit_errors == 0

        # Check that all data has passed through the channel.
        assert counter.calls == 1
        assert counter.count == 2

        counter.reset()

        # Test automatic fragmentation.
        length = 2 * MAX_CHUNK_SIZE + 1
        bit_errors = system(length)

        # Check that no errors have been reported.
        assert bit_errors == 0

        # Check that the data has been split in 3.
        assert counter.calls == 3
        assert counter.count == length


class TestIntegration:
    energy_sensor = EnergySensor()

    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    def test_bpsk_over_awgn(self, eb_n0: float):
        LENGTH = 10**6
        N0 = calculate_n0(eb_n0, ModulatorBPSK.bits_per_symbol)

        config = (ModulatorBPSK(), self.energy_sensor, AWGN(N0), DemodulatorBPSK())
        system = build_system(PseudoRandomStream(), config)

        bit_errors = system(LENGTH)

        # Check with the theoretical rate.
        theoretical_ber = calculate_awgn_ber_with_bpsk(np.asarray(eb_n0))
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

        # All symbols should have unit energy.
        assert np.isclose(self.energy_sensor.mean, 1)

    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    def test_qpsk_over_awgn(self, eb_n0: float):
        LENGTH = 10**6
        N0 = calculate_n0(eb_n0, ModulatorQPSK.bits_per_symbol)

        config = (ModulatorQPSK(), self.energy_sensor, AWGN(N0), DemodulatorQPSK())
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
        LENGTH = 10**6
        N0 = calculate_n0(eb_n0, Modulator16QAM.bits_per_symbol)

        config = (Modulator16QAM(), self.energy_sensor, AWGN(N0), Demodulator16QAM())
        system = build_system(PseudoRandomStream(), config)

        bit_errors = system(LENGTH)

        # Check with the theoretical rate.
        theoretical_ser = calculate_awgn_ser_with_qam(16, np.asarray(eb_n0))
        theoretical_ber = theoretical_ser / 4
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

        # All symbols should have unit energy.
        assert np.isclose(self.energy_sensor.mean, 1, atol=1e-3)

    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    @pytest.mark.parametrize("samples_per_symbol", [2, 4, 8, 16])
    def test_16qam_with_pulse_shaping(self, eb_n0: float, samples_per_symbol: int):
        LENGTH = 2**16
        N0 = calculate_n0(eb_n0, Modulator16QAM.bits_per_symbol)

        config = (
            Modulator16QAM(),
            PulseFilter(up=samples_per_symbol),
            AWGN(N0),
            PulseFilter(down=samples_per_symbol),
            Demodulator16QAM(),
        )
        system = build_system(PseudoRandomStream(), config)

        bit_errors = system(LENGTH)

        # Check with the theoretical rate.
        theoretical_ser = calculate_awgn_ser_with_qam(16, np.asarray(eb_n0))
        theoretical_ber = theoretical_ser / 4
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

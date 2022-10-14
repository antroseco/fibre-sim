import numpy as np
import pytest
from channel import AWGN
from data_stream import PseudoRandomStream
from modulation import DemodulatorBPSK, DemodulatorQPSK, ModulatorBPSK, ModulatorQPSK
from system import build_system
from utils import Component, calculate_awgn_ber_with_bpsk


class Counter(Component):
    input_type = "u8 data"
    output_type = "u8 data"

    def __init__(self) -> None:
        super().__init__()

        self.calls = 0
        self.count = 0

    def __call__(self, data: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.count += data.size

        return data

    def reset(self) -> None:
        self.calls = 0
        self.count = 0


class TestSystem:
    def test_typecheck_system(self):
        # TODO
        pass

    def test_build_system(self):
        counter = Counter()
        data_stream = PseudoRandomStream(1)

        # Test simple data passthrough.
        system = build_system(data_stream, (counter,))

        bit_errors, symbol_errors = system(2)

        # Check that no errors have been reported.
        assert bit_errors == 0
        assert symbol_errors == 0

        # Check that all data has passed through the channel.
        assert counter.calls == 1
        assert counter.count == 2

        counter.reset()

        # Test automatic fragmentation.
        bit_errors, symbol_errors = system(2 * 10**6 + 1)

        # Check that no errors have been reported.
        assert bit_errors == 0
        assert symbol_errors == 0

        # Check that the data has been split in 3.
        assert counter.calls == 3
        assert counter.count == 2 * 10**6 + 1


class TestIntegration:
    @staticmethod
    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    def test_bpsk_over_awgn(eb_n0: float):
        LENGTH = 10**6
        N0 = 1 / eb_n0

        config = (ModulatorBPSK(), AWGN(N0), DemodulatorBPSK())
        system = build_system(PseudoRandomStream(1), config)

        bit_errors, symbol_errors = system(LENGTH)

        # With BPSK, each symbol error corresponds to only one bit error.
        assert bit_errors == symbol_errors

        # Check with the theoretical rate.
        theoretical_ber = calculate_awgn_ber_with_bpsk(np.asarray(eb_n0))
        assert np.isclose(bit_errors / LENGTH, theoretical_ber, rtol=0.05)

    @staticmethod
    @pytest.mark.parametrize("eb_n0", [1, 2, 3])
    def test_qpsk_over_awgn(eb_n0: float):
        LENGTH = 10**6
        # FIXME: converting between Eb and Es is confusing.
        es_n0 = eb_n0 * 2
        N0 = 1 / es_n0

        config = (ModulatorQPSK(), AWGN(N0), DemodulatorQPSK())
        system = build_system(PseudoRandomStream(1), config)

        bit_errors, symbol_errors = system(LENGTH)

        # With QPSK, we can have more than one bit error per symbol.
        # TODO: estimate how many more and add bounds to this check.
        assert bit_errors > symbol_errors

        # Check with the theoretical rate.
        # QPSK bit error rate is equal to the BPSK bit error rate.
        theoretical_ber = calculate_awgn_ber_with_bpsk(np.asarray(eb_n0))
        assert np.isclose(bit_errors / LENGTH / 2, theoretical_ber, rtol=0.05)

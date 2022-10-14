from data_stream import PseudoRandomStream
from system import build_system
from utils import Component

import numpy as np


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

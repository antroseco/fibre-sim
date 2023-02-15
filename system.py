from functools import reduce
from typing import Callable, Optional, Sequence

from numpy.typing import NDArray

from data_stream import DataStream
from filters import PulseFilter
from modulation import Demodulator, Modulator
from phase_recovery import DecisionDirected
from utils import Component, clamp, is_power_of_2, next_power_of_2, row_size

# Very short sequences probably won't work very well.
MIN_CHUNK_SIZE = 2**12  # 4,096

# This is significantly faster than larger/smaller settings. The current
# bottleneck is the circular convolution (using fft/ifft) which is sensitive to
# the length of the data.
MAX_CHUNK_SIZE = 2**16  # 65,536


def typecheck_system(data_stream: DataStream, components: Sequence[Component]) -> None:
    assert data_stream.output_type == components[0].input_type
    assert components[-1].output_type == data_stream.input_type

    for a, b in zip(components, components[1:]):
        assert a.output_type == b.input_type


def check_pulse_filters(components: Sequence[Component]) -> bool:
    count = 0
    first_pf: Optional[PulseFilter] = None

    for component in components:
        if not isinstance(component, PulseFilter):
            continue

        if first_pf:
            assert component.SPAN == first_pf.SPAN
            assert component.BETA == first_pf.BETA
        else:
            first_pf = component

        count += 1

    assert count in (0, 2)

    return count > 0


def check_modulation(components: Sequence[Component]) -> int:
    modulator: Optional[Modulator] = None
    demodulator: Optional[Demodulator] = None

    for component in components:
        # Type matching in case does not instantiate new objects for comparison.
        match component:
            case Modulator():
                assert modulator is None
                modulator = component
            case Demodulator():
                assert demodulator is None
                demodulator = component
            case DecisionDirected():
                assert demodulator is None
                demodulator = component.demodulator

    assert modulator is not None
    assert demodulator is not None

    assert modulator.bits_per_symbol == demodulator.bits_per_symbol

    return modulator.bits_per_symbol


def build_system(
    data_stream: DataStream,
    components: Sequence[Component],
    inspector: Optional[Callable[[str, NDArray], None]] = None,
) -> Callable[[int], int]:
    # FIXME need to update everything to work with the "cd electric field" type.
    # typecheck_system(data_stream, components)

    has_pf = check_pulse_filters(components)
    bits_per_symbol = check_modulation(components)

    def reduce_fn(x: NDArray, component: Component) -> NDArray:
        y = component(x)

        # Call inspector in between components.
        if inspector:
            inspector(type(component).__name__, y)

        # FFT is much more efficient for block sizes that are powers of 2.
        # Enforce this constraint to catch bugs, with a "heuristic".
        length = row_size(y)
        assert is_power_of_2(length) or next_power_of_2(length) - length < 300

        return y

    def simulate_system(bit_count: int) -> int:
        assert bit_count >= MIN_CHUNK_SIZE

        # If we don't reset the data stream, we'll count errors from a previous
        # simulation of the same system.
        data_stream.reset()

        while bit_count >= MIN_CHUNK_SIZE:
            # We'll probably simulate more bits than asked for, but that's fine.
            chunk_size = clamp(bit_count, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)

            if has_pf:
                chunk_size = (
                    PulseFilter.symbols_for_total_length(
                        next_power_of_2(chunk_size // bits_per_symbol)
                    )
                    * bits_per_symbol
                )

            channel_output = reduce(
                reduce_fn, components, data_stream.generate(chunk_size)
            )
            data_stream.validate(channel_output)

            bit_count -= chunk_size

        return data_stream.bit_errors

    return simulate_system

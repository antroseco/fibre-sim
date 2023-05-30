from functools import reduce
from typing import Callable, Optional, Sequence, Type

from numpy.typing import NDArray

from data_stream import DataStream
from filters import PulseFilter
from modulation import Demodulator, Modulator
from phase_recovery import DecisionDirected
from utils import (
    Component,
    Signal,
    TypeChecked,
    clamp,
    is_power_of_2,
    next_power_of_2,
    row_size,
)

# Very short sequences probably won't work very well.
MIN_CHUNK_SIZE = 2**12  # 4,096

# This is significantly faster than larger/smaller settings. The current
# bottleneck is the circular convolution (using fft/ifft) which is sensitive to
# the length of the data.
MAX_CHUNK_SIZE = 2**16  # 65,536


def typecheck_impl(
    left: tuple[Signal, Type, Optional[int]], right: tuple[Signal, Type, Optional[int]]
) -> bool:
    l_sig, l_dtype, l_sps = left
    r_sig, r_dtype, r_sps = right

    if l_sig != r_sig:
        return False

    if l_dtype != r_dtype:
        return False

    if l_sps is None or r_sps is None:
        return True

    return l_sps == r_sps


def typecheck_pair(left: TypeChecked, right: TypeChecked) -> bool:
    return typecheck_impl(left.output_type, right.input_type)


def typecheck_system(data_stream: DataStream, components: Sequence[Component]) -> None:
    assert typecheck_pair(data_stream, components[0])
    assert typecheck_pair(components[-1], data_stream)

    assert all(map(typecheck_pair, components, components[1:]))


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
) -> Callable[[int], float]:
    typecheck_system(data_stream, components)

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
        assert is_power_of_2(length) or next_power_of_2(length) - length < 800

        return y

    def simulate_system(bit_count: int) -> float:
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

        return data_stream.ber

    return simulate_system

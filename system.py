from functools import reduce
from typing import Callable, Optional, Sequence

from numpy.typing import NDArray

from data_stream import DataStream
from utils import Component

# This is significantly faster than larger/smaller settings. The current
# bottleneck is the circular convolution (using fft/ifft) which is sensitive to
# the length of the data.
MAX_CHUNK_SIZE = 2**16  # 65,536


def typecheck_system(data_stream: DataStream, components: Sequence[Component]) -> None:
    assert data_stream.output_type == components[0].input_type
    assert components[-1].output_type == data_stream.input_type

    for a, b in zip(components, components[1:]):
        assert a.output_type == b.input_type


def build_system(
    data_stream: DataStream,
    components: Sequence[Component],
    inspector: Optional[Callable[[str, NDArray], None]] = None,
) -> Callable[[int], int]:
    # FIXME need to update everything to work with the "cd electric field" type.
    # typecheck_system(data_stream, components)

    def reduce_fn(x: NDArray, component: Component) -> NDArray:
        y = component(x)

        # Call inspector in between components.
        if inspector:
            inspector(type(component).__name__, y)

        return y

    def simulate_system(bit_count: int) -> int:
        # If we don't reset the data stream, we'll count errors from a previous
        # simulation of the same system.
        data_stream.reset()

        while bit_count:
            chunk_size = min(bit_count, MAX_CHUNK_SIZE)

            channel_output = reduce(
                reduce_fn, components, data_stream.generate(chunk_size)
            )
            data_stream.validate(channel_output)

            bit_count -= chunk_size

        return data_stream.bit_errors

    return simulate_system

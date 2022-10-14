from functools import reduce
from typing import Callable, Sequence

from data_stream import DataStream
from utils import Component


def typecheck_system(data_stream: DataStream, components: Sequence[Component]) -> None:
    assert data_stream.output_type == components[0].input_type
    assert components[-1].output_type == data_stream.input_type

    for a, b in zip(components, components[1:]):
        assert a.output_type == b.input_type


def build_system(
    data_stream: DataStream, components: Sequence[Component]
) -> Callable[[int], int]:
    typecheck_system(data_stream, components)

    def simulate_system(symbol_count: int) -> int:
        MAX_CHUNK_SIZE = 10**6

        while symbol_count:
            chunk_size = min(symbol_count, MAX_CHUNK_SIZE)

            channel_output = reduce(
                lambda data, comp: comp(data),
                components,
                data_stream.generate(chunk_size),
            )
            data_stream.validate(channel_output)

            symbol_count -= chunk_size

        return data_stream.bit_errors

    return simulate_system

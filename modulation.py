from abc import abstractmethod
from typing import Optional, Type

import numpy as np
from numpy.typing import NDArray

from laser import Laser
from utils import (
    Component,
    Signal,
    bits_to_ints,
    has_one_polarization,
    has_two_polarizations,
    ints_to_bits,
    is_even,
    row_size,
    signal_power,
)


class Modulator(Component):
    bits_per_symbol: int = 0

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.BITS, np.bool_, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 1

    @abstractmethod
    def __call__(self, data: NDArray[np.bool_]) -> NDArray[np.cdouble]:
        assert data.dtype == np.bool_
        assert data.size % self.bits_per_symbol == 0


class Demodulator(Component):
    bits_per_symbol: int = 0

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 1

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.BITS, np.bool_, None

    @abstractmethod
    def __call__(
        self, symbols: NDArray[np.cdouble], scale: Optional[float] = None
    ) -> NDArray[np.bool_]:
        assert symbols.dtype == np.cdouble


class ModulatorBPSK(Modulator):
    bits_per_symbol = 1

    def __call__(self, data: NDArray[np.bool_]) -> NDArray[np.cdouble]:
        super().__call__(data)
        assert has_one_polarization(data)

        # Map bits to symbols (1 -> -1, 0 -> 1).
        return 1 - 2 * data.astype(np.cdouble)


class DemodulatorBPSK(Demodulator):
    bits_per_symbol = 1

    def __call__(
        self, symbols: NDArray[np.cdouble], scale: Optional[float] = None
    ) -> NDArray[np.bool_]:
        super().__call__(symbols)
        assert has_one_polarization(symbols)

        return symbols < 0


class ModulatorQPSK(Modulator):
    bits_per_symbol = 2

    def __call__(self, data: NDArray[np.bool_]) -> NDArray[np.cdouble]:
        super().__call__(data)
        assert has_one_polarization(data)

        # Constellation has 4 symbols. Adjacent symbols only vary by 1 bit.
        # XXX this is compatible with MATLAB's QPSK constellation:
        # >> pskmod([0 1 2 3], 4) * exp(1j * pi/4)
        #  0.7071 + 0.7071i
        # -0.7071 + 0.7071i
        #  0.7071 - 0.7071i
        # -0.7071 - 0.7071i
        #
        #         Q
        #   01    |    00
        #         |
        #  ---------------I
        #         |
        #   11    |    10
        #         |
        in_phase = 1 - 2 * data[1::2]
        quadrature = 1j - 2j * data[0::2]

        # Normalize symbol energy.
        return (in_phase + quadrature) / np.sqrt(2)


class DemodulatorQPSK(Demodulator):
    bits_per_symbol = 2

    def __call__(
        self, symbols: NDArray[np.cdouble], scale: Optional[float] = None
    ) -> NDArray[np.bool_]:
        super().__call__(symbols)
        assert has_one_polarization(symbols)

        # In-phase component is the LSB.
        lsb = np.real(symbols) < 0
        # Quadrature component is the MSB.
        msb = np.imag(symbols) < 0

        data = np.empty(msb.size + lsb.size, dtype=np.bool_)
        data[0::2] = msb
        data[1::2] = lsb

        return data


class DemodulatorDQPSK(DemodulatorQPSK):
    @staticmethod
    def gray_to_binary(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return (array ^ (array >> 1)).astype(np.uint8, copy=False)

    def __call__(
        self, symbols: NDArray[np.cdouble], scale: Optional[float] = None
    ) -> NDArray[np.bool_]:
        assert has_one_polarization(symbols)

        bits = super().__call__(symbols, scale)
        assert is_even(bits.size)

        # This is simpler if we convert each bit pair into an int and then take
        # their difference. Because the constellation is Gray coded, we need to
        # convert to normal binary first. Fortunately, that's straightforward.
        gray = bits_to_ints(bits, 2)

        ints = self.gray_to_binary(gray.astype(np.uint8, copy=False))

        # Differential coding: data is stored in the phase difference between
        # symbols. First symbol is assumed to be 0, to match MATLAB (but should
        # be discarded).
        decoded = np.diff(ints, prepend=np.uint8(0)) & 0b11
        assert decoded.dtype == np.uint8

        return ints_to_bits(decoded, 2)


class Modulator16QAM(Modulator):
    bits_per_symbol = 4

    @staticmethod
    def impl(msbs: NDArray[np.bool_], lsbs: NDArray[np.bool_]) -> NDArray[np.int64]:
        assert msbs.size == lsbs.size

        # Looking at the two MSBs of the constellation symbols, we can see that
        # they completly determine the in-phase component. Looking at the two
        # LSBs, we can see that they determine the quadrature component. As they
        # follow the same Gray code each time (00 -> 01 -> 11 -> 10), we only
        # need to look at two bits at a time to determine each component.
        #
        # This function only needs to convert from Gray code to binary. It is
        # straightforward to derive the following equation (only valid for 2
        # bits) by tabulating the Gray-coded inputs next to the binary output.
        return (msbs << 1) | (msbs ^ lsbs)

    def __call__(self, data: NDArray[np.bool_]) -> NDArray[np.cdouble]:
        super().__call__(data)
        assert has_one_polarization(data)

        # Constellation has 16 symbols. Adjacent symbols only vary by 1 bit.
        # 1101 is at (1, 1) and 1000 is at (3, 3).
        #
        # There are multiple possible constellations. This one is identical to
        # the one generated by MATLAB's qammod(x, 16, 'gray'), to ensure
        # compatibility.
        #
        #              Q
        #   0000  0100 | 1100  1000
        #              |
        #              |
        #   0001  0101 | 1101  1001
        #              |
        #   ------------------------I
        #              |
        #   0011  0111 | 1111  1011
        #              |
        #              |
        #   0010  0110 | 1110  1010
        #              |
        #
        # In-phase component carries the 2 MSBs.
        in_phase = -3 + 2 * self.impl(data[0::4], data[1::4])
        # Quadrature component carries the 2 LSBs.
        quadrature = 3j - 2j * self.impl(data[2::4], data[3::4])

        # Normalize symbol energy. The mean energy of the constellation is
        # defined as the expected value of |a+bj|^2. With a uniform
        # distribution over all symbols, this comes out to 10. Thus, we need to
        # divide the *amplitude* by the square root of 10.
        return (in_phase + quadrature) / np.sqrt(10)


class Demodulator16QAM(Demodulator):
    bits_per_symbol = 4

    @staticmethod
    def impl(
        symbols: NDArray[np.float64], scale: float
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        # FIXME explanation. Replace magic numbers.
        msbs = symbols > 0
        lsbs = np.abs(symbols) <= (2 * scale / np.sqrt(10))

        return msbs, lsbs

    def __call__(
        self, symbols: NDArray[np.cdouble], scale: Optional[float] = None
    ) -> NDArray[np.bool_]:
        super().__call__(symbols)
        assert has_one_polarization(symbols)

        # FIXME this should be replaced with a proper filter. Due to pulse
        # shaping and downsampling, the symbols no longer have unit energy. As
        # we rely on a threshold to distinguish between the inner and outer
        # constellation squares, we need to scale it based on the mean energy
        # of the received symbols.
        if scale is None:
            scale = np.sqrt(signal_power(symbols))

        # Make the type checker happy.
        assert scale is not None

        # Each symbols carries 4 bits. The in-phase component contains the 2
        # MSBs, and the quadrature component contains the 2 LSBs.
        data = np.empty(symbols.size * self.bits_per_symbol, dtype=np.bool_)
        data[0::4], data[1::4] = self.impl(np.real(symbols), scale)
        data[2::4], data[3::4] = self.impl(-np.imag(symbols), scale)

        return data


class IQModulator(Component):
    def __init__(self, laser: Laser) -> None:
        super().__init__()

        self.laser = laser

    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, None

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.OPTICAL, np.cdouble, None

    @staticmethod
    def iq_impl(
        voltages: NDArray[np.cdouble], laser_output: NDArray[np.cdouble]
    ) -> NDArray[np.cdouble]:
        assert has_one_polarization(voltages)
        assert has_one_polarization(laser_output)
        assert voltages.size == laser_output.size

        def max_a(x: NDArray[np.float64]) -> float:
            return np.max(np.abs(x))

        # Input voltage should range from -Vπ to +Vπ. Remove this restriction
        # from the caller by inferring Vπ.
        Vpi = max(max_a(np.real(voltages)), max_a(np.imag(voltages)))

        # FIXME increase Vpi to make the cosine function look more linear. Is
        # there a better way to do this?
        Vpi *= 1.35

        # Add the DC bias of -Vπ, and divide by Vπ, which is 1.
        phi = (voltages - Vpi) * (np.pi / Vpi)

        Efield = 0.5 * laser_output
        Efield *= np.cos(np.real(phi) / 2) + 1j * np.sin(np.imag(phi) / 2)

        return Efield

    def __call__(self, voltages: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        return self.iq_impl(voltages, self.laser(voltages.size))


class DPModulator(IQModulator):
    def __call__(self, voltages: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_two_polarizations(voltages)

        laser_output = self.laser(row_size(voltages))

        pol_v = 0.5 * self.iq_impl(voltages[0], laser_output)
        pol_h = 0.5 * self.iq_impl(voltages[1], laser_output)

        return np.vstack((pol_v, pol_h))


class AlamoutiEncoder(Component):
    @property
    def input_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 1

    @property
    def output_type(self) -> tuple[Signal, Type, Optional[int]]:
        return Signal.SYMBOLS, np.cdouble, 1

    def __call__(self, data: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        assert has_one_polarization(data)
        # NOTE symbols must come in pairs.
        assert is_even(data.size)

        pol_x = np.empty_like(data)
        pol_y = np.empty_like(data)

        pol_x[0::2] = data[0::2]
        pol_x[1::2] = -np.conj(data[1::2])

        pol_y[0::2] = data[1::2]
        pol_y[1::2] = np.conj(data[0::2])

        return np.vstack((pol_x, pol_y))

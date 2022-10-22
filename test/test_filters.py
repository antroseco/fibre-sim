import numpy as np
import pytest
from filters import Upsampler

# TODO
# test for unit energy
# test for pulse bandwidth
# test for perfect recovery in the absence of noise
# test against the spectrum/series given for the raised cosine


class TestUpsampler:
    @staticmethod
    @pytest.mark.parametrize("factor", range(2, 5))
    def test_upsampling(factor: int):
        data = np.arange(4) + np.arange(4) * 1j

        upsampler = Upsampler(factor)
        upsampled = upsampler(data)

        # Check array properties.
        assert upsampled.ndim == data.ndim
        assert upsampled.size == factor * data.size
        assert upsampled.dtype == data.dtype
        assert np.all(np.isfinite(upsampled))

        # Check structure.
        indices = np.arange(0, upsampled.size, factor, dtype=int)
        assert np.all(upsampled[indices] == data)
        assert np.all(upsampled[~indices] == 0)

        # Check that energy is preserved.
        assert np.sum(np.abs(upsampled) ** 2) == np.sum(np.abs(data) ** 2)

    @staticmethod
    @pytest.mark.parametrize("factor", range(-1, 2))
    def test_factor(factor: int):
        with pytest.raises(Exception):
            Upsampler(factor)

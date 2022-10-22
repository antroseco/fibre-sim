import numpy as np
import pytest
from filters import Upsampler, root_raised_cosine

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


class TestRootRaisedCosine:
    @staticmethod
    def test_low_beta():
        BETA = 0.11
        SPAN = 6
        SAMPLES_PER_SYMBOL = 4

        # Test against MATLAB's rcosdesign().
        EXPECTED = (
            -0.0136,
            0.0260,
            0.0570,
            0.0559,
            0.0144,
            -0.0509,
            -0.1024,
            -0.0985,
            -0.0149,
            0.1388,
            0.3188,
            0.4631,
            0.5181,
            0.4631,
            0.3188,
            0.1388,
            -0.0149,
            -0.0985,
            -0.1024,
            -0.0509,
            0.0144,
            0.0559,
            0.0570,
            0.0260,
        )

        result = root_raised_cosine(BETA, SAMPLES_PER_SYMBOL, SPAN)

        assert np.all(np.isfinite(result))
        assert result.ndim == 1
        assert result.size == SAMPLES_PER_SYMBOL * SPAN
        assert np.allclose(result, EXPECTED, atol=0.0001)

    @staticmethod
    def test_high_beta():
        BETA = 0.99
        SPAN = 6
        SAMPLES_PER_SYMBOL = 4

        # Test against MATLAB's rcosdesign().
        EXPECTED = (
            -0.0045,
            0.0004,
            0.0064,
            -0.0006,
            -0.0103,
            0.0006,
            0.0182,
            -0.0013,
            -0.0432,
            0.0013,
            0.2141,
            0.5000,
            0.6353,
            0.5000,
            0.2141,
            0.0013,
            -0.0432,
            -0.0013,
            0.0182,
            0.0006,
            -0.0103,
            -0.0006,
            0.0064,
            0.0004,
        )

        result = root_raised_cosine(BETA, SAMPLES_PER_SYMBOL, SPAN)

        assert np.all(np.isfinite(result))
        assert np.all(np.isreal(result))
        assert result.ndim == 1
        assert result.size == SAMPLES_PER_SYMBOL * SPAN
        assert np.allclose(result, EXPECTED, atol=0.0001)

    @staticmethod
    @pytest.mark.parametrize("beta", (-0.1, 0, 1, 1.1))
    def test_invalid_beta(beta: float):
        with pytest.raises(Exception):
            root_raised_cosine(beta, 4, 4)

    @staticmethod
    @pytest.mark.parametrize("span", (-1, 0, 1, 3, 9))
    def test_invalid_span(span: int):
        with pytest.raises(Exception):
            root_raised_cosine(0.1, 4, span)

    @staticmethod
    @pytest.mark.parametrize("samples_per_symbol", (-1, 0, 1, 3, 9))
    def test_invalid_samples_per_symbol(samples_per_symbol: int):
        with pytest.raises(Exception):
            root_raised_cosine(0.1, samples_per_symbol, 4)

    @staticmethod
    @pytest.mark.parametrize("beta", (0.01, 0.22, 0.51, 0.78, 0.99))
    @pytest.mark.parametrize("samples_per_symbol", (2, 4, 16))
    @pytest.mark.parametrize("span", (2, 4, 16))
    def test_energy(beta: float, samples_per_symbol: int, span: int):
        result = root_raised_cosine(beta, samples_per_symbol, span)
        assert np.all(np.isreal(result))

        energy = np.sum(result**2)
        assert np.isclose(energy, 1, rtol=1e-3)

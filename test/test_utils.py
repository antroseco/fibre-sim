import numpy as np
import pytest
from matplotlib import pyplot as plt
from utils import calculate_awgn_ser_with_qam, plot_ber, calculate_awgn_ber_with_bpsk


class TestPlotBer:
    def test_data_and_labels_length(self):
        _, ax = plt.subplots()

        # Must fail when ber and labels have different lengths.
        with pytest.raises(Exception):
            plot_ber(ax, (), (), ("", ""))

        # This is fine.
        plot_ber(ax, (), ((), ()), ("", ""))


class TestCalculateAwgnBerWithBpsk:
    @staticmethod
    def test_monotonically_decreasing():
        x = np.linspace(1, 10, 20)
        y = calculate_awgn_ber_with_bpsk(x)

        assert np.all(np.diff(y) < 0)

    @staticmethod
    def test_known_values():
        assert np.allclose(
            calculate_awgn_ber_with_bpsk(np.asarray((4, 9))),
            (0.00233886749, 0.00001104524),
        )


class TestCalculateAwgnSerWithQam:
    @staticmethod
    def test_monotonically_decreasing():
        x = np.linspace(1, 10, 20)
        y = calculate_awgn_ser_with_qam(16, x)

        assert np.all(np.diff(y) < 0)

    @staticmethod
    def test_known_values():
        assert np.allclose(
            calculate_awgn_ser_with_qam(16, np.asarray((4, 9))),
            (0.11045740518, 0.01093553713),
        )

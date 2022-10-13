import numpy as np
import pytest
from matplotlib import pyplot as plt
from utils import plot_ber, calculate_awgn_ber_with_bpsk


class TestPlotBer:
    def test_data_and_labels_length(self):
        _, ax = plt.subplots()

        # Must fail when ber and labels have different lengths.
        with pytest.raises(Exception):
            plot_ber(ax, (), (), ("", ""))

        # This is fine.
        plot_ber(ax, (), ((), ()), ("", ""))


class TestCalculateAwgnBerWithBpsk:
    def test_monotonically_decreasing(self):
        x = np.linspace(1, 10, 20)
        y = calculate_awgn_ber_with_bpsk(x)

        assert np.all(np.diff(y) < 0)

    def test_known_values(self):
        assert np.allclose(
            calculate_awgn_ber_with_bpsk(np.asarray((4, 9))),
            (0.00233886749, 0.00001104524),
        )

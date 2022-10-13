import pytest
from utils import plot_ber

from matplotlib import pyplot as plt


class TestPlotBer:
    def test_data_and_labels_length(self):
        _, ax = plt.subplots()

        # Must fail when ber and labels have different lengths.
        with pytest.raises(Exception):
            plot_ber(ax, (), (), ("", ""))

        # This is fine.
        plot_ber(ax, (), ((), ()), ("", ""))

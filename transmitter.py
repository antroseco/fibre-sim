import numpy as np
from numpy.typing import NDArray

from utils import Component, power_dbm_to_lin, signal_power


class Transmitter(Component):
    input_type = "cd electric field"
    output_type = "cd symbols"

    def __init__(self, tx_power_dbm: float) -> None:
        super().__init__()

        self.tx_power = power_dbm_to_lin(tx_power_dbm)

    def __call__(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        # Set transmitted signal power to that specified.
        print(signal_power(signal), self.tx_power)
        return signal * np.sqrt(self.tx_power / signal_power(signal))

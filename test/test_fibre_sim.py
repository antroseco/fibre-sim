import numpy as np
from fibre_sim import simulate
from utils import calculate_awgn_ber_with_bpsk


class TestSimulation:
    def test_simulate(self):
        LENGTH = 10**6

        eb_n0 = np.linspace(1, 3, 5)

        simulation = [simulate(LENGTH, 1 / i)[0] / LENGTH for i in eb_n0]
        theoretical = calculate_awgn_ber_with_bpsk(eb_n0)

        assert np.allclose(simulation, theoretical, rtol=0.05)

# Auto controller: energy-shaping swing-up → LQR balance handoff.
#
# While the pole is below 140°, an energy-shaping law pumps the
# pendulum up.  Once it crosses 140° the LQR takes over and holds
# it inverted.

import numpy as np
from numpy import sin, cos, pi
from physics import G, R, M, m

rad = pi / 180


class LQRController:
    name     = "Auto — LQR"
    controls = []   # fully automatic, no manual inputs

    # ------------------------------------------------------------------ #
    def get_force(self, q):
        """Return the cart force (N) given state q = [theta, x, dtheta, dx]."""
        if q[0] < 135 * rad:
            return self._swing_up(q)
        else:
            return self._lqr(q)

    def process_event(self, _event):
        """No keyboard / mouse interaction needed."""
        pass

    # ------------------------------------------------------------------ #
    def _swing_up(self, q):
        """Energy-shaping controller that pumps the pendulum upward."""
        Ee    = 0.5 * m * R**2 * q[2]**2 - m * G * R * (1 + cos(q[0]))
        k     = 0.23
        A     = k * Ee * cos(q[0]) * q[2]
        delta = m * sin(q[0])**2 + M
        return (A * delta
                - m * R * q[2]**2 * sin(q[0])
                - m * G * sin(q[0]) * cos(q[0]))

    def _lqr(self, q):
        """LQR gain vector tuned for the inverted equilibrium (theta = pi)."""
        # k1, k2, k3, k4 = 134.003, -2.236, 38.896, -6.924  # balanced preset
        k1, k2, k3, k4 = 174.541, -10.000, 51.984, -16.325  # super_aggressive preset

        return -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])

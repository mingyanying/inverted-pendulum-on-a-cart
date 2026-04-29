# Auto controller: energy-shaping swing-up → PID balance handoff.
#
# Same swing-up as LQR, but once near the top a PID loop on the pole
# angle takes over instead of the optimal LQR gain vector.
#
# Gains to tune:
#   Kp, Ki, Kd  — angle PID  (error = theta - pi)
#   Kp_x, Kd_x — position PD (keeps cart near centre)

import numpy as np
from numpy import sin, cos, pi
from physics import G, R, M, m, U_MAX

rad = pi / 180

Kp   = 120.0   # proportional on angle error
Ki   =   0.0   # integral on angle error  (corrects slow drift)
Kd   =  35.0   # derivative on angle      (= angular velocity damping)
Kp_x = -10.0   # proportional on cart position (negative: pull back to centre)
Kd_x =  -8.0   # derivative on cart velocity


class PIDController:
    name     = "Auto — PID"
    controls = []

    def __init__(self):
        self._integral  = 0.0
        self._dt        = 1 / 60   # matches simulation timestep

    # ------------------------------------------------------------------ #
    def get_force(self, q):
        if q[0] < 140 * rad:
            self._integral = 0.0   # reset integrator while swinging up
            return self._swing_up(q)
        return self._pid(q)

    def process_event(self, _event):
        pass

    # ------------------------------------------------------------------ #
    def _swing_up(self, q):
        """Energy-shaping pump (identical to LQRController)."""
        Ee    = 0.5 * m * R**2 * q[2]**2 - m * G * R * (1 + cos(q[0]))
        k     = 0.23
        A     = k * Ee * cos(q[0]) * q[2]
        delta = m * sin(q[0])**2 + M
        return (A * delta
                - m * R * q[2]**2 * sin(q[0])
                - m * G * sin(q[0]) * cos(q[0]))

    def _pid(self, q):
        """PID on pole angle + PD on cart position."""
        e_theta         = q[0] - pi          # angle error
        self._integral += e_theta * self._dt
        # anti-windup: clamp integral so it can't build up unboundedly
        if Ki != 0.0:
            self._integral = np.clip(self._integral, -U_MAX / Ki, U_MAX / Ki)

        u = -(Kp   * e_theta
            + Ki   * self._integral
            + Kd   * q[2]            # theta_dot
            + Kp_x * q[1]            # cart position
            + Kd_x * q[3])           # cart velocity
        return u

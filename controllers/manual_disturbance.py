# Manual controller: full swing-up + LQR balance, with user disturbances.
#
# The controller swings the pole up and balances it automatically.
# Once it is balancing you can give the ball a small nudge to test recovery:
#
#   SPACE      → tiny kick to theta_dot (alternates direction each press)
#   Left click → kick in the direction of the click
#
# Kicks are silently ignored while the pole is still swinging up.

import numpy as np
from numpy import sin, cos, pi
import pygame
from physics import G, R, M, m

KICK_STRENGTH  = 1    # rad/s  — small nudge
BALANCE_THRESH = 30      # degrees from vertical; kicks only work inside this

rad = pi / 180


class DisturbanceController:
    name     = "Manual — disturbance"
    controls = [
        "[SPACE]   nudge ball (while balancing)",
        "[L-click] directional nudge",
    ]

    def __init__(self):
        self._pending_kick = 0.0
        self._kick_sign    = 1.0   # alternates each SPACE press

    # ------------------------------------------------------------------ #
    def get_force(self, q):
        """Swing-up while pole is low, LQR once it is near the top."""
        if q[0] < 140 * rad:
            return self._swing_up(q)
        return self._lqr(q)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self._pending_kick += self._kick_sign * KICK_STRENGTH
            self._kick_sign    *= -1.0   # alternate direction

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            screen_w = pygame.display.get_surface().get_width()
            sign = -1.0 if event.pos[0] < screen_w // 2 else 1.0
            self._pending_kick += sign * KICK_STRENGTH

    def consume_impulse(self, q):
        """
        Return the kick (rad/s) to add to theta_dot, then clear it.
        Returns 0 if the pole is not yet balancing near the top.

        Call in the main loop before the physics step:
            q[2] += controller.consume_impulse(q)
            q     = rk4_step(q, ...)
        """
        near_top = abs(q[0] - pi) < BALANCE_THRESH * rad
        kick, self._pending_kick = self._pending_kick, 0.0
        return kick if near_top else 0.0

    # ------------------------------------------------------------------ #
    def _swing_up(self, q):
        Ee    = 0.5 * m * R**2 * q[2]**2 - m * G * R * (1 + cos(q[0]))
        k     = 0.23
        A     = k * Ee * cos(q[0]) * q[2]
        delta = m * sin(q[0])**2 + M
        return (A * delta
                - m * R * q[2]**2 * sin(q[0])
                - m * G * sin(q[2]) * cos(q[2]))

    def _lqr(self, q):
        k1, k2, k3, k4 = 140.560, -3.162, 41.772, -8.314
        return -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])

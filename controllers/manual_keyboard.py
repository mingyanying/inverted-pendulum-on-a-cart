# Manual controller: keyboard left / right.
#
# Hold LEFT  arrow → push cart left   (negative force)
# Hold RIGHT arrow → push cart right  (positive force)
# Release         → no force (cart coasts)
#
# The force magnitude is tunable via FORCE_N.

import pygame

FORCE_N = 10.0   # Newtons applied while a key is held


class KeyboardController:
    name     = "Manual — keyboard"
    controls = [
        "[←]      push cart left",
        "[→]      push cart right",
    ]

    # ------------------------------------------------------------------ #
    def get_force(self, _q):
        """
        Poll the current key state every frame.
        Returns the net cart force (N).
        """
        keys = pygame.key.get_pressed()
        force = 0.0
        if keys[pygame.K_LEFT]:
            force -= FORCE_N
        if keys[pygame.K_RIGHT]:
            force += FORCE_N
        return force

    def process_event(self, _event):
        """Key polling happens in get_force(); nothing to do here."""
        pass

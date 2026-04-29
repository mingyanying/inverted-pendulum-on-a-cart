import sys, os
import numpy as np
from numpy import cos, pi
import pygame

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from physics import R, X_LIM, G, M, m, rk4_step
from controllers import (
    LQRController,
    PIDController,
    ManualController,
    KeyboardController,
    DisturbanceController,
)

# ---------------------------------------------------------------------------
# Pygame setup
# ---------------------------------------------------------------------------
def setup_pygame():
    pygame.init()
    W, H   = 960, 600
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Cart-Pole")
    clock  = pygame.time.Clock()
    font_l = pygame.font.SysFont("Arial", 20, bold=True)
    font_s = pygame.font.SysFont("Arial", 15)
    return screen, clock, font_l, font_s, W, H


def make_coords(W, H):
    SCALE      = 130
    SCR_CX     = W // 2
    SCR_PY     = int(H * 0.62)
    CART_W_PX  = int(0.40 * SCALE)
    CART_H_PX  = int(0.20 * SCALE)
    WHEEL_R_PX = max(int(0.06 * SCALE), 4)
    BALL_R_PX  = max(int(0.09 * SCALE), 4)

    def to_px(sx, sy):
        return int(SCR_CX + sx * SCALE), int(SCR_PY - sy * SCALE)

    return to_px, CART_W_PX, CART_H_PX, WHEEL_R_PX, BALL_R_PX


def make_colours():
    return dict(
        BG      = ( 20,  20,  30),
        RAIL    = (160, 160, 160),
        LIMIT   = (210,  50,  50),
        CART    = ( 60, 120, 200),
        CART_E  = (180, 210, 255),
        WHEEL   = ( 70,  70,  70),
        WHEEL_E = (130, 130, 130),
        POLE    = (255, 140,   0),
        BALL    = (255, 210,  60),
        PIVOT   = (220, 220, 220),
        TEXT    = (220, 220, 220),
        ANGLE   = ( 80, 220, 120),
        SHADOW  = ( 40,  40,  50),
        ACTIVE  = (100, 230, 100),
        DIM     = (110, 110, 120),
        CTRL    = (140, 180, 255),
        PANEL   = (  0,   0,   0, 150),
    )


# ---------------------------------------------------------------------------
# Session recorder — logs force + state + energies for the first 30 s
# ---------------------------------------------------------------------------
RECORD_SECS = 30

class Recorder:
    def __init__(self):
        self._t   = []
        self._u   = []
        self._q   = []   # [theta, x, theta_dot, x_dot] per frame

    def record(self, t, q, u):
        if t > RECORD_SECS:
            return
        self._t.append(t)
        self._u.append(u)
        self._q.append(q.copy())

    def plot(self):
        if len(self._t) < 2:
            return
        t  = np.array(self._t)
        u  = np.array(self._u)
        q  = np.array(self._q)          # shape (N, 4)

        theta     = q[:, 0]
        x         = q[:, 1]
        theta_dot = q[:, 2]
        x_dot     = q[:, 3]

        KE   = 0.5 * m * R**2 * theta_dot**2
        PE   = m * G * R * (1 - cos(theta))          # 0 at bottom, 2mgR at top
        E    = KE + PE                                # total mechanical energy
        E_err = KE - m * G * R * (1 + cos(theta))    # energy error from upright (0 = balanced)

        fig = plt.figure(figsize=(13, 9))
        fig.suptitle(f"Session recording  ({t[-1]:.1f} s,  {len(t)} frames)",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

        # ── Force ────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, :])
        ax.plot(t, u, color="#F44336", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.fill_between(t, u, alpha=0.15, color="#F44336")
        ax.set_title("Cart force  u (N)", fontweight="bold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)")
        ax.grid(True, alpha=0.3)

        # ── Angle ────────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t, theta * 180 / pi, color="#4CAF50", linewidth=1.2)
        ax2.axhline(180, color="gray", linewidth=0.6, linestyle="--", label="upright (180°)")
        ax2.set_title("Pole angle  θ (°)")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("θ (degrees)")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        # ── Cart position ────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t, x, color="#2196F3", linewidth=1.2)
        ax3.axhline(0, color="gray", linewidth=0.6, linestyle="--", label="centre")
        ax3.set_title("Cart position  x (m)")
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("x (m)")
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

        # ── Angular & cart velocity ───────────────────────────────────────
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(t, theta_dot * 180 / pi, color="#FF9800", linewidth=1.1, label="θ̇ (°/s)")
        ax4.plot(t, x_dot,                color="#9C27B0", linewidth=1.1, label="ẋ (m/s)", linestyle="--")
        ax4.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax4.set_title("Velocities")
        ax4.set_xlabel("Time (s)"); ax4.set_ylabel("velocity")
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

        # ── Energies ──────────────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(t, KE,    color="#FF9800", linewidth=1.1, label="KE pendulum")
        ax5.plot(t, PE,    color="#4CAF50", linewidth=1.1, label="PE pendulum")
        ax5.plot(t, E,     color="#2196F3", linewidth=1.4, label="Total E")
        ax5.plot(t, E_err, color="#F44336", linewidth=1.0, label="E error (swing-up)", linestyle="--")
        ax5.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax5.set_title("Mechanical energies (J)")
        ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Energy (J)")
        ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

        plt.savefig(os.path.join(os.path.dirname(__file__), "models", "session_plot.png"),
                    dpi=150, bbox_inches="tight")
        print("[Recorder] Plot saved to models/session_plot.png")
        plt.show()


# ---------------------------------------------------------------------------
# HUD panel
# ---------------------------------------------------------------------------
PANEL_W = 240

def draw_hud(screen, font_s, C, controllers, current, W):
    """Draw a semi-transparent panel listing all controllers and active bindings."""
    lines = []

    # --- controller list ---
    lines.append(("header", "── Controllers ──"))
    for key_char, ctrl in controllers.items():
        active  = ctrl is current
        colour  = C["ACTIVE"] if active else C["DIM"]
        prefix  = "▶" if active else " "
        lines.append(("ctrl", f" {prefix} [{key_char}]  {ctrl.name}", colour))

    # --- mode-specific controls ---
    mode_controls = getattr(current, "controls", [])
    if mode_controls:
        lines.append(("spacer", ""))
        lines.append(("header", "── Controls ──"))
        for line in mode_controls:
            lines.append(("bind", line))

    # --- always-on manual controls ---
    lines.append(("spacer", ""))
    lines.append(("header", "── Always on ──"))
    for bind in [
        "[←] [→]  push cart",
        "[SPACE]  nudge ball",
        "[click]  directional nudge",
    ]:
        lines.append(("bind", bind))

    # --- global controls ---
    lines.append(("spacer", ""))
    lines.append(("header", "── Global ──"))
    for bind in ["[R]      Reset", "[ESC]    Quit"]:
        lines.append(("bind", bind))

    # measure height
    line_h  = 20
    padding = 12
    total_h = padding * 2 + len(lines) * line_h
    panel_x = W - PANEL_W - 12
    panel_y = 12

    # draw translucent panel
    surf = pygame.Surface((PANEL_W, total_h), pygame.SRCALPHA)
    surf.fill(C["PANEL"])
    pygame.draw.rect(surf, (80, 80, 100, 180), surf.get_rect(), 1)
    screen.blit(surf, (panel_x, panel_y))

    # draw text rows
    y = panel_y + padding
    for entry in lines:
        kind = entry[0]
        text = entry[1]
        if kind == "spacer":
            y += line_h // 2
            continue
        elif kind == "header":
            colour = C["DIM"]
            font   = font_s
        elif kind == "ctrl":
            colour = entry[2]
            font   = font_s
        else:                    # "bind"
            colour = C["CTRL"]
            font   = font_s

        surf_text = font.render(text, True, colour)
        screen.blit(surf_text, (panel_x + padding, y))
        y += line_h


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_frame(screen, font_l, font_s, C, to_px,
               CART_W_PX, CART_H_PX, WHEEL_R_PX, BALL_R_PX,
               q, t_elapsed, controllers, controller, W):
    theta, x = q[0], q[1]
    cart_sx  = x
    ball_sx  = R * np.sin(theta) + x
    ball_sy  = -R * cos(theta)

    screen.fill(C["BG"])

    # rail + end-stops
    _, cy_cart     = to_px(0, 0)
    rail_y         = cy_cart + CART_H_PX + WHEEL_R_PX * 2
    lim_lx         = to_px(-X_LIM, 0)[0]
    lim_rx         = to_px( X_LIM, 0)[0]
    STOP_W, STOP_H = 10, 28

    pygame.draw.line(screen, C["RAIL"], (lim_lx, rail_y), (lim_rx, rail_y), 4)
    for stop in (
        (lim_lx - STOP_W, rail_y - STOP_H + 4, STOP_W, STOP_H),
        (lim_rx,           rail_y - STOP_H + 4, STOP_W, STOP_H),
    ):
        pygame.draw.rect(screen, C["LIMIT"], stop)
        pygame.draw.rect(screen, (255, 120, 120), stop, 2)

    screen.blit(font_s.render("",True, C["LIMIT"]),
                (lim_lx - STOP_W - 2, rail_y - STOP_H - 16))
    screen.blit(font_s.render("",True, C["LIMIT"]),
                (lim_rx + 4,           rail_y - STOP_H - 16))

    # cart shadow
    cx_s, cy_s = to_px(cart_sx, 0.0)
    pygame.draw.rect(screen, C["SHADOW"],
                     (cx_s - CART_W_PX//2 + 4, cy_s - CART_H_PX + 4,
                      CART_W_PX, CART_H_PX))

    # cart body
    cart_rect = (cx_s - CART_W_PX//2, cy_s - CART_H_PX, CART_W_PX, CART_H_PX)
    pygame.draw.rect(screen, C["CART"],   cart_rect)
    pygame.draw.rect(screen, C["CART_E"], cart_rect, 2)

    # wheels
    wy = cy_s + WHEEL_R_PX
    for wx in (cx_s - CART_W_PX//4, cx_s + CART_W_PX//4):
        pygame.draw.circle(screen, C["WHEEL"],   (wx, wy), WHEEL_R_PX)
        pygame.draw.circle(screen, C["WHEEL_E"], (wx, wy), WHEEL_R_PX, 2)

    # pole + ball
    bx_s, by_s = to_px(ball_sx, ball_sy)
    pygame.draw.line(screen, C["POLE"], (cx_s, cy_s), (bx_s, by_s), 5)
    pygame.draw.circle(screen, C["BALL"],       (bx_s, by_s), BALL_R_PX)
    pygame.draw.circle(screen, (255, 255, 255), (bx_s, by_s), BALL_R_PX, 2)

    # pivot dot
    pygame.draw.circle(screen, C["PIVOT"], (cx_s, cy_s), 5)

    # state readout (top-left)
    screen.blit(font_l.render(f"t = {t_elapsed:.1f} s",        True, C["TEXT"]),  (20, 20))
    screen.blit(font_l.render(f"θ = {theta * 180/pi:.1f}°",    True, C["ANGLE"]), (20, 48))
    screen.blit(font_s.render(f"x = {x:.3f} m",                True, C["TEXT"]),  (20, 76))
    screen.blit(font_s.render(f"ω = {q[2]*180/pi:.1f} °/s",   True, C["TEXT"]),  (20, 96))

    # controller panel (top-right)
    draw_hud(screen, font_s, C, controllers, controller, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DT            = 1 / 60
INITIAL_STATE = np.array([0.0, 0.0, 0.1, 0.0])   # [theta, x, theta_dot, x_dot]

# Auto controllers (switchable with number keys)
controllers = {
    "1": LQRController(),
    "2": PIDController(),
    "3": ManualController(),
}

KEY_MAP = {
    pygame.K_1: "1",
    pygame.K_2: "2",
    pygame.K_3: "3",
}

# Always-on manual add-ons (active regardless of which auto controller is running)
keyboard    = KeyboardController()
disturbance = DisturbanceController()

screen, clock, font_l, font_s, W, H = setup_pygame()
to_px, CART_W_PX, CART_H_PX, WHEEL_R, BALL_R = make_coords(W, H)
C = make_colours()

controller = controllers["1"]
q          = INITIAL_STATE.copy()
t_elapsed  = 0.0
running    = True
recorder   = Recorder()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                q, t_elapsed = INITIAL_STATE.copy(), 0.0
            elif event.key in KEY_MAP:
                controller = controllers[KEY_MAP[event.key]]

        # Pass events to active controller and manual add-ons
        controller.process_event(event)
        disturbance.process_event(event)

    # Auto controller force + keyboard override always on top
    u  = controller.get_force(q)
    u += keyboard.get_force(q)

    # Disturbance kick (only fires when near the top)
    q[2] += disturbance.consume_impulse(q)

    recorder.record(t_elapsed, q, u)
    q = rk4_step(q, u, DT)
    t_elapsed += DT

    draw_frame(screen, font_l, font_s, C,
               to_px, CART_W_PX, CART_H_PX, WHEEL_R, BALL_R,
               q, t_elapsed, controllers, controller, W)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
recorder.plot()
sys.exit()

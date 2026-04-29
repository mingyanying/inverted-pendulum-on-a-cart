# Inverted Pendulum on a Cart — PHY 329

Real-time simulation of a cart-pole system with automatic swing-up and balance
controllers, rendered in Pygame.

```
        O  ← ball  (m = 1 kg)
       /
      /  ← pole  (R = 1 m)
     /
 [cart]  ← cart  (M = 4 kg)
──────────────────────────  ← rail  (±3 m)
```

[![Demo](https://img.youtube.com/vi/xosSy-7pbzY/maxresdefault.jpg)](https://youtu.be/xosSy-7pbzY)

---

## Quick start

```bash
pip install -r requirements.txt
python balance_pygame.py
```

Press **[1]**, **[2]**, or **[3]** to switch controllers at any time.

---

## Project structure

```
inverted-pendulum-on-a-cart/
├── balance_pygame.py        # simulation window (Pygame)
├── physics.py               # equations of motion + RK4 integrator
├── tune_gains.py            # compute LQR / PID gains offline
├── requirements.txt
├── controllers/
│   ├── auto_lqr.py          # energy-shaping swing-up → LQR balance
│   ├── auto_pid.py          # energy-shaping swing-up → PID balance
│   ├── manual_keyboard.py   # always-on keyboard cart push
│   ├── manual_disturbance.py# always-on nudge / disturbance tool
│   └── manual_only.py       # no-force passthrough
└── models/
    └── session_plot.png     # auto-generated on quit (gitignored)
```

---

## Physics (`physics.py`)

### State vector

```
q = [theta, x, theta_dot, x_dot]
```

| Variable    | Meaning                       | Unit  |
|-------------|-------------------------------|-------|
| `theta`     | pole angle from straight-down | rad   |
| `x`         | cart position                 | m     |
| `theta_dot` | pole angular velocity         | rad/s |
| `x_dot`     | cart velocity                 | m/s   |

`theta = 0` → hanging straight down. `theta = π` → inverted (balance target).

### Physical constants

| Symbol   | Value  | Meaning             |
|----------|--------|---------------------|
| `m`      | 1 kg   | ball mass           |
| `M`      | 4 kg   | cart mass           |
| `R`      | 1 m    | pole length         |
| `G`      | 9.8    | gravity (m/s²)      |
| `U_MAX`  | 50 N   | max actuator force  |
| `X_LIM`  | 3 m    | rail half-length    |
| `B_CART` | 0.3    | cart drag (N·s/m)   |
| `B_POLE` | 0.05   | pole drag (N·m·s/rad)|

Each frame advances the simulation with a single **RK4 step** at `dt = 1/60 s`.
Cart speed is capped at `V_MAX = 5 m/s` and position is hard-clamped to `±X_LIM`.

---

## Controllers

All controllers share the same interface:

```python
u = controller.get_force(q)    # returns force in Newtons
controller.process_event(event)
```

### [1] LQR (`auto_lqr.py`)

**Swing-up** (`theta < 135°`) — energy-shaping controller. Measures the energy
error from the upright position and pumps or removes energy to drive the pole up:

```
E_error = ½mR²θ̇² − mgR(1 + cosθ)    (= 0 at top, standing still)
A       = k · E_error · cosθ · θ̇
u       = A·δ − mR·θ̇²·sinθ − mg·sinθ·cosθ
δ       = m·sin²θ + M
```

**Balance** (`theta ≥ 135°`) — Linear Quadratic Regulator with a precomputed
gain vector:

```
u = −(k1·(θ−π) + k2·x + k3·θ̇ + k4·ẋ)
k = [174.541, −10.000, 51.984, −16.325]   (super_aggressive preset)
```

Gains are computed offline by solving the continuous-time LQR Algebraic Riccati
Equation for the linearised system at the upright equilibrium.

### [2] PID (`auto_pid.py`)

Same energy-shaping swing-up as LQR. Once `theta ≥ 140°`, a PID loop takes over:

```
u = −( Kp   × (θ − π)
      + Ki   × ∫(θ−π) dt
      + Kd   × θ̇
      + Kp_x × x
      + Kd_x × ẋ )
```

| Gain    | Value  | Role                              |
|---------|--------|-----------------------------------|
| `Kp`    | 120.0  | angle proportional                |
| `Ki`    |   0.0  | angle integral (anti-windup)      |
| `Kd`    |  35.0  | angle derivative / angular damping|
| `Kp_x`  | −10.0  | cart position (return to centre)  |
| `Kd_x`  |  −8.0  | cart velocity damping             |

### [3] Manual (`manual_only.py`)

No automatic force — the cart is driven entirely by the always-on keyboard
and disturbance inputs listed below.

---

## Always-on controls

These are active regardless of which controller is selected:

| Input          | Effect                                      |
|----------------|---------------------------------------------|
| `[←]` / `[→]` | Push cart left / right (10 N)               |
| `[SPACE]`      | Nudge the ball (only works near balance)    |
| `[Left click]` | Directional nudge toward the click side     |
| `[R]`          | Reset simulation to initial state           |
| `[ESC]`        | Quit (saves a session plot to `models/`)    |

---

## Tuning gains (`tune_gains.py`)

Computes optimal gains and optionally writes them back into the controller files.

```bash
python tune_gains.py                          # print gains only
python tune_gains.py --apply                  # write into controller files
python tune_gains.py --apply --lqr-preset aggressive
python tune_gains.py --lqr-only               # skip PID optimisation
python tune_gains.py --pid-only               # skip LQR
```

**LQR** — solved analytically via the continuous-time Algebraic Riccati Equation.
Three presets: `balanced`, `aggressive`, `super_aggressive`.

**PID** — tuned numerically with Nelder-Mead, minimising a simulation cost
(integral of squared errors + force penalty) over random near-upright starts.

---

## Design notes

**Why energy-shaping for swing-up?**
The large-angle swing-up is a nonlinear problem. PID and LQR are linearised
controllers that only work near the upright equilibrium. Energy-shaping works
globally by targeting the correct total mechanical energy regardless of angle.

**Why LQR?**
LQR gives the minimum-control-effort linear stabiliser for a given trade-off
between state error and actuator effort (encoded in the Q/R matrices). The gain
vector is computed once offline and applied as a simple dot product at runtime.

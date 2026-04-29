"""
Compute optimal LQR gains and tune PID gains for the cart-pole system.

LQR  — solved analytically via the Algebraic Riccati Equation (ARE).
         Three Q/R presets are offered: balanced, aggressive, smooth.

PID  — tuned numerically by minimising a simulation cost function
         (integral of squared errors + force penalty) over many random
         near-upright starting conditions.

Usage:
    python tune_gains.py                        # compute and print only
    python tune_gains.py --apply                # also write into controller files
    python tune_gains.py --apply --lqr-preset aggressive
"""
import sys, os, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numpy import pi
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize

from physics import G, R, M, m, B_CART, B_POLE, U_MAX, X_LIM, rk4_step

DT = 1 / 60


# Linearised system around upright equilibrium 
#
# Let φ = θ − π  (small deviation from upright).
# Linearising the equations of motion at (θ=π, x=0, θ̇=0, ẋ=0):
#
#   sin(θ) ≈ −φ,  cos(θ) ≈ −1,  δ = m·sin²θ + M ≈ M
#
# Gives state-space form:  ẋ = A·x + B·u   with x = [φ, x, φ̇, ẋ]ᵀ

A = np.array([
    [0,                  0,  1,                                    0           ],
    [0,                  0,  0,                                    1           ],
    [(m+M)*G / (M*R),    0,  -(m+M)*B_POLE / (m*M*R**2),         -B_CART/(M*R)],
    [m*G / M,            0,  -B_POLE / (M*R),                     -B_CART/M   ],
])

B = np.array([[0], [0], [1/(M*R)], [1/M]])


# ── LQR ───────────────────────────────────────────────────────────────────

# Each preset is (Q_diag, R_scalar).
# Q weights: [angle error, cart position, angular velocity, cart velocity]
# R weights:  control effort
LQR_PRESETS = {
    "balanced":   (np.diag([50.0,  1.0,  5.0,  1.0]), 1.0),
    "aggressive": (np.diag([200.0, 5.0,  20.0, 2.0]), 0.5),
    "smooth":     (np.diag([20.0,  0.5,  2.0,  0.5]), 2.0),
}


def lqr_gains(Q, R_cost):
    """Solve the continuous-time ARE and return the gain vector K (4,)."""
    P = solve_continuous_are(A, B, Q, np.array([[R_cost]]))
    K = (1.0 / R_cost) * (B.T @ P)
    return K.flatten()


def run_lqr():
    """Compute and print gains for all presets. Returns dict of results."""
    print("\n── LQR  (Algebraic Riccati Equation) ──────────────────────────")
    print(f"  Linearisation point: θ=π (upright),  x=0,  θ̇=0,  ẋ=0")
    print(f"  State:  [θ−π,  x,  θ̇,  ẋ]")
    print(f"  Force:  u = −(k1·(θ−π) + k2·x + k3·θ̇ + k4·ẋ)\n")
    print(f"  {'Preset':12s}  {'k1 (θ−π)':>10}  {'k2 (x)':>10}  "
          f"{'k3 (θ̇)':>10}  {'k4 (ẋ)':>10}")
    print("  " + "─" * 56)

    results = {}
    for name, (Q, R_cost) in LQR_PRESETS.items():
        K = lqr_gains(Q, R_cost)
        results[name] = K
        print(f"  {name:12s}  {K[0]:10.3f}  {K[1]:10.3f}  "
              f"{K[2]:10.3f}  {K[3]:10.3f}")

    current = np.array([140, -3, 41, -8])
    print(f"  {'(current)':12s}  {current[0]:10.3f}  {current[1]:10.3f}  "
          f"  {current[2]:10.3f}  {current[3]:10.3f}")
    return results


# ── PID cost function ──────────────────────────────────────────────────────

def _pid_cost(gains, n_trials=12, n_steps=900, seed=0):
    """
    Simulate from n_trials random near-upright starts, return mean cost.

    Cost per step:
      100·φ²  +  1·x²  +  0.1·θ̇²  +  0.1·ẋ²  +  0.01·(u/U_MAX)²

    A large flat penalty is added if the pole falls (|φ| > 90°).
    """
    Kp, Ki, Kd, Kp_x, Kd_x = gains

    # Soft penalty for gains leaving reasonable ranges
    bound_pen = (max(0, -Kp)      * 1e4 +
                 max(0, -Ki)      * 1e4 +
                 max(0, -Kd)      * 1e4 +
                 max(0,  Kp_x)    * 1e4 +    # Kp_x must be ≤ 0
                 max(0,  Kd_x)    * 1e4)      # Kd_x must be ≤ 0
    if bound_pen > 0:
        return 1e8 + bound_pen

    rng = np.random.default_rng(seed)
    total = 0.0

    for _ in range(n_trials):
        q = np.array([
            pi + rng.uniform(-0.15, 0.15),
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.15, 0.15),
        ])
        integral = 0.0
        ep_cost  = 0.0

        for _ in range(n_steps):
            phi       = q[0] - pi
            integral += phi * DT
            Ki_safe   = abs(Ki) + 1e-9
            integral  = np.clip(integral, -U_MAX / Ki_safe, U_MAX / Ki_safe)

            u = -(Kp   * phi
                + Ki   * integral
                + Kd   * q[2]
                + Kp_x * q[1]
                + Kd_x * q[3])

            ep_cost += (100.0 * phi**2
                      +   1.0 * q[1]**2
                      +   0.1 * q[2]**2
                      +   0.1 * q[3]**2
                      +   0.01 * (u / U_MAX)**2)

            q = rk4_step(q, u, DT)

            if abs(q[0] - pi) > pi / 2:       # pole fell — abort episode
                ep_cost += 1e6
                break

        total += ep_cost

    return total / n_trials


def run_pid():
    """Numerically optimise PID gains. Returns gain array (5,)."""
    x0 = np.array([120.0, 8.0, 35.0, -3.0, -8.0])   # current gains

    print("\n── PID  (Nelder-Mead simulation optimisation) ─────────────────")
    print("  Optimising: Kp, Ki, Kd, Kp_x, Kd_x")
    print(f"  Starting cost (current gains): {_pid_cost(x0):.1f}")
    print("  Running optimiser ...\n")

    result = minimize(
        _pid_cost, x0,
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 0.2, "fatol": 10.0, "disp": True},
    )

    K = result.x
    cost_orig = _pid_cost(x0)
    cost_opt  = result.fun
    improve   = (cost_orig - cost_opt) / cost_orig * 100

    print(f"\n  ── Results ───────────────────────────────────")
    print(f"  {'Gain':8s}  {'Original':>10}  {'Optimised':>10}")
    print(f"  {'─'*32}")
    labels  = ["Kp", "Ki", "Kd", "Kp_x", "Kd_x"]
    for lbl, orig, opt in zip(labels, x0, K):
        print(f"  {lbl:8s}  {orig:10.2f}  {opt:10.2f}")
    print(f"\n  Cost  original : {cost_orig:,.1f}")
    print(f"  Cost  optimised: {cost_opt:,.1f}  ({improve:+.1f}%)")

    return K


# ── Apply to files ─────────────────────────────────────────────────────────

def apply_lqr(K, preset):
    path = os.path.join("controllers", "auto_lqr.py")
    with open(path) as f:
        src = f.read()
    src = re.sub(
        r'k1, k2, k3, k4 = [^\n]+',
        f'k1, k2, k3, k4 = {K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}, {K[3]:.3f}',
        src,
    )
    with open(path, "w") as f:
        f.write(src)
    print(f"  controllers/auto_lqr.py  ← {preset} preset: "
          f"[{K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}, {K[3]:.3f}]")


def apply_pid(K):
    path = os.path.join("controllers", "auto_pid.py")
    with open(path) as f:
        src = f.read()

    pairs = [
        (r'(Kp\s*=\s*)\S+(\s*#)',   K[0]),
        (r'(Ki\s*=\s*)\S+(\s*#)',   K[1]),
        (r'(Kd\s*=\s*)\S+(\s*#)',   K[2]),
        (r'(Kp_x\s*=\s*)\S+(\s*#)', K[3]),
        (r'(Kd_x\s*=\s*)\S+(\s*#)', K[4]),
    ]
    for pattern, val in pairs:
        src = re.sub(pattern, lambda m, v=val: f"{m.group(1)}{v:.2f}{m.group(2)}", src)

    with open(path, "w") as f:
        f.write(src)
    labels = ["Kp", "Ki", "Kd", "Kp_x", "Kd_x"]
    vals   = "  ".join(f"{l}={v:.2f}" for l, v in zip(labels, K))
    print(f"  controllers/auto_pid.py  ← {vals}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute optimal LQR and PID gains")
    p.add_argument("--apply", action="store_true",
                   help="Write optimised gains into controller files")
    p.add_argument("--lqr-preset", choices=list(LQR_PRESETS), default="aggressive",
                   help="LQR preset to apply when --apply is used (default: balanced)")
    p.add_argument("--pid-only",   action="store_true", help="Skip LQR, run PID only")
    p.add_argument("--lqr-only",   action="store_true", help="Skip PID, run LQR only")
    args = p.parse_args()

    lqr_results = None
    pid_gains   = None

    if not args.pid_only:
        lqr_results = run_lqr()

    if not args.lqr_only:
        pid_gains = run_pid()

    if args.apply:
        print("\n── Writing to controller files ────────────────────────────────")
        if lqr_results is not None:
            apply_lqr(lqr_results[args.lqr_preset], args.lqr_preset)
        if pid_gains is not None:
            apply_pid(pid_gains)
        print("\n  Done. Restart the simulation to use the new gains.")
    else:
        print("\n  (Run with --apply to write gains into the controller files.)")

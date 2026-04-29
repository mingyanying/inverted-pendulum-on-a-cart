# Shared physical constants and equations of motion.
# Imported by balance_pygame.py and all controllers.

import numpy as np
from numpy import sin, cos

# Physical parameters
G     = 9.8   # gravity          (m/s^2)
R     = 1.0   # pole length      (m)
M     = 4.0   # cart mass        (kg)
m     = 1.0   # ball mass        (kg)
X_LIM = 2  # rail half-length (m)

# Air resistance (linear drag)
B_CART = 0.3    # cart translational drag  (N·s/m)
B_POLE = 0.05   # pole/ball rotational drag (N·m·s/rad)

# Cart hardware limits
U_MAX  = 50.0   # max actuator force       (N)
V_MAX  = 5.0    # max cart speed           (m/s)


def equations_of_motion(q, u):
    """
    Cart-pole equations of motion with air resistance.

    q = [theta, x, theta_dot, x_dot]
        theta     : pole angle from straight-down (rad)
        x         : cart position (m)
        theta_dot : angular velocity (rad/s)
        x_dot     : cart velocity (m/s)

    u : horizontal force applied to cart (N)

    Drag is derived from the Lagrangian so the coupling terms are exact:
      θ̈ drag contribution : -(m+M)·B_POLE/(m·R²)·θ̇  +  cos(θ)·B_CART·ẋ/R
      ẍ drag contribution : -B_CART·ẋ                 +  cos(θ)·B_POLE·θ̇/R
    (all divided by delta = m·sin²θ + M)

    Returns dq/dt as a numpy array.
    """
    theta, x, dtheta, dx = q
    delta   = m * sin(theta)**2 + M
    dqdt    = np.zeros(4)
    dqdt[0] = dtheta
    dqdt[1] = dx
    dqdt[2] = (- m * dtheta**2 * sin(theta) * cos(theta)
               - (m + M) * G * sin(theta) / R
               - u * cos(theta) / R
               - (m + M) * B_POLE / (m * R**2) * dtheta
               + cos(theta) * B_CART * dx / R) / delta
    dqdt[3] = (  m * R * dtheta**2 * sin(theta)
               + m * G * sin(theta) * cos(theta)
               + u
               - B_CART * dx
               + cos(theta) * B_POLE * dtheta / R) / delta
    return dqdt


def apply_wall(q):
    """
    Hard-wall constraint: clamp cart to [-X_LIM, X_LIM] and
    zero out velocity that would push it further into the wall.
    Returns a new state array (does not mutate q).
    """
    q = q.copy()
    if q[1] > X_LIM:
        q[1] = X_LIM
        q[3] = min(q[3], 0.0)
    elif q[1] < -X_LIM:
        q[1] = -X_LIM
        q[3] = max(q[3], 0.0)
    return q


"""Single RK4 integration step with force saturation, velocity cap, and wall constraint."""

def rk4_step(q, u, dt):
    
    u = np.clip(u, -U_MAX, U_MAX)          # saturate actuator force
    k1 = equations_of_motion(q,            u)
    k2 = equations_of_motion(q + dt/2*k1,  u)
    k3 = equations_of_motion(q + dt/2*k2,  u)
    k4 = equations_of_motion(q + dt*k3,    u)
    q_new = q + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    q_new[3] = np.clip(q_new[3], -V_MAX, V_MAX)   # cap cart speed
    return apply_wall(q_new)

import numpy as np
import scipy.io as sio
from numpy.linalg import norm, inv, cond

def dual_point(x):
    """Evaluate the dual point"""
    z = A @ x - b
    v = 2 * z
    maxA_v = norm(A.T @ v, ord=np.inf)
    if maxA_v > lambd:
        v = v * lambd / maxA_v
    return v


def G(v):
    """Evaluate the dual objective"""
    return -0.25 * v.T @ v - v.T @ b


def duality_gap(x, G):
    """Evaluate the duality gap"""
    return objective(x) - G


def update_t(t, nu, s, mu=2, s_min=0.5):
    """Update rule for log-barrier parameter t.

    Keyword arguments:
    t -- current value of t
    nu --  duality gap
    s -- step size
    mu -- update parameter (G.P. ratio), > 0 (default 2)
    s_min -- update parameter in (0,1] (default 0.5)

    Returns:
    t -- updated t
    """

    if s > s_min:
        return max(mu * min(2 * n / nu, t), t)
    else:
        return t


def dual_int_point(x0, u0, epsilon=0.001):
    """Dual Newton interior point method (DNIPM) with backtracking line search.

    Keyword arguments:
    x0, u0 -- n vectors; starting point
    nu -- positive scalar; stopping threshold (default 0.001)

    Returns:
    x -- n vector; minimised point
    """

    global t

    assert phi(x0, u0) < np.inf, '(x0, u0) not in feasible region'

    x, u = x0, u0
    count = 0
    v = dual_point(x)
    nu = duality_gap(x, G(v))

    # Evaluate stopping criterion
    while (nu / G(v)) > epsilon:
        # Newton descent direction
        dphi = - inv(hes_phi(x, u)) @ grad_phi(x, u)
        # Step size through backtracking l.s
        step_size = backtrack_ls(x, u, dphi)

        # Update (x,u) -> (x+dx,u+du)
        x = x + step_size * dphi[:n]
        u = u + step_size * dphi[n:]

        # Evaluate the duality gap
        v = dual_point(x)
        nu = duality_gap(x, G(v))

        # Update t
        t = update_t(t, nu, step_size, mu=2)

        count += 1
        print('Iteration: {} | Objective: {} | Stopping Crit: {}'.format(count, objective(x), nu / G(v)))
        print('----------------------------------------------------------------------------------')
    return x


# Set initial log-barrier accuracy parameter
t = 1 / lambd

# DNIPM
x = dual_int_point(x0, u0, epsilon=0.001)

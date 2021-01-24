import numpy as np
import scipy.io as sio
from numpy.linalg import norm, inv, pinv, cond

# Load problem data
A, x0 = sio.loadmat('A.mat'), sio.loadmat('x0.mat')
A, x0 = A['A'], x0['x']
b = (A @ x0).flatten()

m, n = A.shape[0], A.shape[1]


def objective(x):
    """Evaluate the original (l1-regularised least squares problem) objective function"""
    return norm(A @ x - b, ord=2) ** 2 + lambd * norm(x, ord=1)


def Phi(x, u):
    """Evaluate the log-barrier function"""
    Phi_1 = np.where(u - x > 0, u - x, 0)
    Phi_2 = np.where(u + x > 0, u + x, 0)
    return - sum(np.log(Phi_1) + np.log(Phi_2))


def phi(x, u):
    """Evaluate the central-path formulation"""
    return t * norm(A @ x - b, ord=2) ** 2 + t * lambd * sum(u) + Phi(x, u)


def grad_phi(x, u):
    """Evaluate the gradient of the central-path formulation"""
    gradx_phi = 2 * t * A.T @ (A @ x - b) + 1 / (u - x) - 1 / (u + x)

    gradu_phi = t * lambd - 1 / (u - x) - 1 / (u + x)

    grad_phi = np.concatenate((gradx_phi, gradu_phi))

    return grad_phi


def hes_phi(x, u):
    """Evaluate the Hessian of the central-path formulation"""
    C = np.diagflat(1 / (u + x) ** 2) + np.diagflat(1 / (u - x) ** 2)
    E = np.diagflat(1 / (u + x) ** 2) - np.diagflat(1 / (u - x) ** 2)
    hes_phi = np.block([[2 * t * A.T @ A + C, E], [E, C]])
    return hes_phi


def backtrack_ls(x, u, dxu, alpha=0.1, beta=0.4):
    """Backtracking linesearch. Finds the step length to approximately minimise
    phi along the ray {(x + step_size * dx, u + step_size * du) | step_size > 0}

    Keyword arguments:
    x, u -- n vectors; point at which to evaluate step size
    dxu -- 2n vector; descent direction
    alpha -- scalar in (0, 0.5); backtrack l.s. constant (default 0.1)
    beta -- scalar in (0, 1); backtrack l.s. constant (default 0.4)

    Returns:
    step_size -- scalar in (0,1); step size by which to move in descent direction dxu
    """

    step_size = 1
    while phi(x + step_size * dxu[:n], u + step_size * dxu[n:]) > phi(x, u) - alpha * step_size * dxu.T @ dxu:
        step_size *= beta
    return step_size


def int_point(x0, u0, eta=1e-10):
    """Newton interior point method (NIPM) with backtracking line search.

    Keyword arguments:
    x0, u0 -- n vectors; starting point
    nu -- positive scalar; stopping threshold (default 1e-10)

    Returns:
    x -- n vector; minimised point
    """

    assert phi(x0, u0) < np.inf, '(x0, u0) not in feasible region'
    x, u = x0, u0
    count = 0
    s = grad_phi(x, u).T @ inv(hes_phi(x, u)) @ grad_phi(x, u)
    # Evaluate stopping criterion
    while s / 2 > eta:
        # Newton descent direction
        dxu = - inv(hes_phi(x, u)) @ grad_phi(x, u)
        step_size = backtrack_ls(x, u, dxu)
        x = x + step_size * dxu[:n]
        u = u + step_size * dxu[n:]
        s = grad_phi(x, u).T @ inv(hes_phi(x, u)) @ grad_phi(x, u)
        count += 1
        print('Iteration: {} | Objective: {} | Stopping Crit: {}'.format(count, objective(x), s / 2))
        print('----------------------------------------------------------------------------------')
    return x


# Set regularisation parameter
lambd_max = max(abs(2 * A.T @ b).flatten())
lambd = 0.01 * lambd_max

# Set log-barrier accuracy parameter
t = 10 ** 6

# Initialise feasible starting point
x0, u0 = np.zeros(n), np.ones(n)

# NIPM
x = int_point(x0, u0, eta=1e-10)

# Investigation into the effect of parameter t
t_array = np.logspace(2, 6, 20)
t_objective = []
for t in t_array:
    x = int_point(x0, u0, eta=1e-10)
    t_objective.append(objective(x))

# Minimum energy reconstruction
x_me = pinv(A) @ b  #pinv(A) returns the Moorse-Penrose Inverse of A

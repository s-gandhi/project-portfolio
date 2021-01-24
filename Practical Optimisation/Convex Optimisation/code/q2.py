import numpy as np
import scipy.io as sio
from numpy.linalg import norm, inv, cond

# Load problem data
A, b = sio.loadmat('A3.mat'), sio.loadmat('b3.mat')
A, b = A['A3'], b['b3']

m, n = A.shape[0], A.shape[1]
I = np.identity(m)
# Construct l1 LP matrices, as per eq. (10)
A_til = np.concatenate((np.concatenate((A, -I), axis=1), np.concatenate((-A, -I), axis=1)), axis=0)
b_til = np.concatenate((b, -b), axis=0)[:, 0]
c_til = np.concatenate((np.zeros(n), np.ones(m)), axis=0)


def phi(x):
    """Evaluate the log-barrier function"""
    r_til = b_til - A_til @ x
    # dom phi = {x | A_til x < b_til}, so replace negative value r_til[i] by 0
    r_til = np.where(r_til > 0, r_til, 0)
    return -np.sum(np.log(r_til))


def grad_phi(x):
    """Evaluate the gradient of the log-barrier function"""
    # From compact notation, grad_phi = A_til^T  d
    d = 1 / (b_til - A_til @ x)
    return A_til.T @ d


def hes_phi(x):
    """Evaluate the Hessian of the log-barrier function"""
    # From compact notation, hes_phi = A_til^T  diag(d^2) A_til^T
    d = 1 / (b_til - A_til @ x)
    return A_til.T @ np.diag(d ** 2) @ A_til


def J(x, t=1):
    """Evaluate the objective function"""
    return t * np.dot(c_til, x) + phi(x)


def grad_J(x, t=1):
    """Evaluate the gradient of objective function"""
    return t * c_til + grad_phi(x)


def hes_J(x, t=1):
    """Evaluate the Hessian of objective function"""
    return hes_phi(x)


def backtrack_desc(x0, alpha=0.1, beta=0.4, nu=1e-3):
    """Gradient descent with backtracking line search.

    Keyword arguments:
    x0 -- 2n vector; starting point
    alpha -- scalar in (0, 0.5); backtrack l.s. constant (default 0.1)
    beta -- scalar in (0, 1); backtrack l.s. constant (default 0.4)
    nu -- positive scalar; stopping threshold (default 1e-3)
    
    Returns:
    x -- 2n vector; minimised point
    J_hist -- list; history data of objective function
    count -- scalar; number of iterations to convergence
    """

    assert phi(x0) < np.inf, 'x0 not in feasible region'
    x = x0
    count = 0
    J_hist = []
    dx = - grad_J(x)
    # Evaluate stopping criterion
    while norm(dx, ord=2) > nu:
        J_hist.append(J(x))
        s = 1
        # Backtrack l.s.
        while J(x + s * dx) > J(x) - alpha * s * dx.T @ dx:
            s *= beta
        x = x + s * dx
        dx = - grad_J(x)
        count += 1
    return x, J_hist, count


def exact_desc(x0, nu=1e-3):
    """Gradient descent with exact line search.

    Keyword arguments:
    x0 -- 2n vector; starting point
    nu -- positive scalar; stopping threshold (default 1e-3)
    
    Returns:
    x -- 2n vector; minimised point
    J_hist -- list; history data of objective function
    count -- scalar; number of iterations to convergence
    """

    assert phi(x0) < np.inf, 'x0 not in feasible region'
    x = x0
    count = 0
    J_eval = []
    dx = - grad_J(x)
    # Evaluate stopping criterion
    while norm(dx, ord=2) > nu:
        J_eval.append(J(x))
        # Exact l.s.
        s = (np.dot(dx, dx)) / (dx.T @ hes_J(x) @ dx)
        x = x + s * dx
        dx = - grad_J(x)
        count += 1
    return x, J_eval, count


# Initial point in feasible region
x0 = (inv(A_til.T @ A_til) @ A_til.T) @ (b_til - 3)

# Perform gradient descent
x_bt, J_eval_bt, count_bt = backtrack_desc(x0)
x_e, J_eval_e, count_e = exact_desc(x0)

# optimised objective (backtrack l.s, exact l.s.)
bt_min_obj = norm((A @ x_bt[:n] - b.flatten()), ord=1)
e_min_obj = norm((A @ x_e[:n] - b.flatten()), ord=1)

# Condition number of Hessian
condition_number = cond(hes_J(x_bt))

# Investigation into the effect of alpha on backtracking l.s
alpha_array = np.arange(0.05, 0.55, 0.05)
alpha_counts = []
for alpha in alpha_array:
    x_bt, J_eval_bt, count_bt = backtrack_desc(x0, alpha=alpha, beta=0.4)
    alpha_counts.append(count_bt)

# Investigation into the effect of beta on backtracking l.s
beta_array = np.arange(0.05, 1, 0.05)
beta_counts = []
for beta in beta_array:
    x_bt, J_eval_bt, count_bt = backtrack_desc(x0, alpha=0.1, beta=beta)
    beta_counts.append(count_bt)

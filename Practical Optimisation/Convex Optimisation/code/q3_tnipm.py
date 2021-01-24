import numpy as np
import scipy.io as sio
from numpy.linalg import norm, inv

def Axfunc_l1(x, A, At, d1, d2, p1, p2, p3):
    """Compute AX (PCG)

    Returns:
    y = hessphi*[x1;x2], where hessphi = [A.T * A * 2 + D1, D2; D2, D1]
    """

    x1 = x[:n]
    x2 = x[n:]
    y1 = (At @ ((A @ x1) * 2)) + np.multiply(d1, x1) + np.multiply(d2, x2)
    y2 = np.multiply(d2, x1) + np.multiply(d1, x2)
    y = np.concatenate((y1, y2))
    return y


def Mfunc_l1(x, A, At, d1, d2, p1, p2, p3):
    """Compute P^{-1}X (PCG):

    Returns:
    y = P^{-1} * x
    """

    x1 = x[:n]
    x2 = x[n:]
    y1 = np.multiply(p1, x1) - np.multiply(p2, x2)
    y2 = -np.multiply(p2, x1) + np.multiply(p3, x2)
    y = np.concatenate((y1, y2))
    return y


def tnipm(x0, u0, epsilon=0.001):
    """Truncated Newton interior point method (TNIPM) with backtracking line search.

    Keyword arguments:
    x0, u0 -- n vectors; starting point
    nu -- positive scalar; stopping threshold (default 0.001)

    Returns:
    x -- n vector; minimised point
    """

    global t

    assert phi(x0, u0) < np.inf, '(x0, u0) not in feasible region'

    x, u = x0, u0
    e_pcg = epsilon
    count = 0

    v = dual_point(x)
    nu = duality_gap(x, G(v))

    # Evaluate stopping criterion
    while (nu / G(v)) > epsilon:

        # Compute the search direction (notation consistent with MATLAB script)
        q1 = 1 / (u + x)
        q2 = 1 / (u - x)

        d1 = t * (q1 ** 2 + q2 ** 2)
        d2 = t * (q1 ** 2 - q2 ** 2)

        diagxtx = np.diag(At @ A)

        prb = diagxtx + d1
        prs = np.multiply(prb, d1) - (d2 ** 2)
        p1, p2, p3 = np.divide(d1, prs), np.divide(d2, prs), np.divide(prb, prs)

        x_to_Ax = lambda x: Axfunc_l1(x, A, At, d1, d2, p1, p2, p3)
        x_to_Mx = lambda x: Mfunc_l1(x, A, At, d1, d2, p1, p2, p3)

        Ax = LinearOperator((2 * n, 2 * n), matvec=x_to_Ax)
        M = LinearOperator((2 * n, 2 * n), matvec=x_to_Mx)

        dxu = cg(Ax, -grad_phi(x, u), tol=e_pcg, M=M)[0]  # PCG approximation to Newton system

        # Compute step size by backtracking l.s.
        step_size = backtrack_ls(x, u, dxu)

        # Update (x,u) -> (x+dx,u+du)
        x = x + step_size * dxu[:n]
        u = u + step_size * dxu[n:]

        # Evaluate the duality gap
        v = dual_point(x)
        nu = duality_gap(x, G(v))

        # Update t, pcg tolerance
        t = update_t(t, nu, step_size, mu=2)
        e_pcg = min(0.1, epsilon * nu / norm(dxu, ord=2))

        count += 1
        if count % 10000 == 0:
            print('Iteration: {} | Objective: {} | Stopping Crit: {}'.format(count, objective(x), nu / G(v)))
            print('----------------------------------------------------------------------------------')
    print(count, objective(x))
    return x

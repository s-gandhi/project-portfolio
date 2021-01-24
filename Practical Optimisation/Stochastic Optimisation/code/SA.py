import numpy as np
from numpy.linalg import norm


def f(x):
    """
    Function which evaluates the Rana-Function for an n-dimensional solution.

    Parameters
    ----------
    x : (n, ) array
        Solution to be evaluated.

    Returns
    -------
    f_eval : float
        Objective function evaluated at the input x.

    """
    # Initialise the output value
    f_eval = 0
    # Summation over (n-1) dimensions
    for i in range(x.shape[0] - 1):
        f_eval += x[i] * np.cos(np.sqrt(np.abs(x[i + 1] + x[i] + 1))) * np.sin(
            np.sqrt(np.abs(x[i + 1] - x[i] + 1))) + (1 + x[i + 1]) * np.cos(
            np.sqrt(np.abs(x[i + 1] - x[i] + 1))) * np.sin(np.sqrt(np.abs(x[i + 1] + x[i] + 1)))
    return f_eval


class SA:
    """
    Simulated annealing optimisation to find the minimum of an objective function f. The initial annealing temperature is found by performing an initial search and selecting a temperature which yields a specified average probability of acceptance for increases in f. The trial solution formula suggested by Parks [1990] takes the form:
    x_{i+1} = x_{i} + Du,
    where u ~ U(-1,1), and D is a diagonal matrix which specifies the maximum change permitted in each variable. D is updated after successful changes to the objective function by folding in information regarding the magnitudes of successful changes made. The temperature is decremented adaptively by the formulation from Huang et al. [1986] once a given number of trials have been performed at the current temperature, or if a minimum number of acceptances have been made, whichever comes first. If a given number of iterations have elapsed without improvement to the current best solution, a restart is performed, and the search resumed from the current best solution. The archive stores the current best solutions, based on a Euclidean distance dissimilarity criterion.
    """

    def __init__(self, f, n, bound, max_iter=10000, seed=0, max_chain=100, chi_0=0.8, alpha=0.1, omega=2.1,
                 d_u=None, d_l=None, cooling_scheme='adaptive', max_restart=2000, chi_f=0.01, archive_size=25,
                 d_min=None, d_sim=None):
        """
        Class attribute initialisation.

        Parameters
        ----------
        f : func
            Objective function to be minimised.
        n : int
            Number of dimensions over which to optimise.
        bound : float
            Symmetric bound (+/-) on the domain of the control variables.
        max_iter : int, optional
            Maximum number of objective function evaluations. The default is 10000.
        seed : int, optional
            Random number generator seed for reproducibility. The default is 0.
        max_chain : int, optional
            Number of iterations after which to terminate Markov chain and decrement temperature. The default is 100.
        chi_0 : float, optional
            Avg probability of accepting increases in f during initial search. The default is 0.8.
        alpha : float, optional
            Damping constant. The default is 0.1.
        omega : float, optional
            Weighting constant. The default is 2.1.
        d_u : float, optional
            Upper limit on the elements of the diagonal matrix D. If d_u is None (default), d_u is set to bound / 5.
        d_l : float, optional
            Lower limit on the elements of the diagonal matrix D. If d_l is None (default), d_l is set to d_u / 1e5.
        cooling_scheme : str, optional
            Choice of cooling scheme, either 'exponential' (Kirkpatrick et al. [1982]) or 'adaptive' (Huang et al. [1986]). The default is 'adaptive'.
        max_restart : int, optional
            Number of iterations after which to restart search from the current best solution. The default is 2000.
        chi_f : float, optional
            Solution acceptance ratio termination threshold. The default is 0.01.
        archive_size : int, optional
            Number of solutions to store in the archive. The default is 25.
        d_min : float, optional
            Archive dissimilarity threshold. If d_min is None (default), d_min is set to bound / 100.
        d_sim : float, optional
            Archive dissimilarity threshold. If d_sim is None (default), d_sim is set to d_min / 100.
        """
        # assertion check
        assert cooling_scheme in ['exponential', 'adaptive'], \
            "cooling scheme must either be 'exponential' or 'adaptive'"
        # initialise attributes
        self.f = f
        self.n = n
        self.bound = np.abs(bound)
        self.max_iter = max_iter
        self.seed = seed
        self.max_chain = max_chain
        self.chi_0 = chi_0
        self.alpha = alpha
        self.omega = omega
        if d_u:
            self.d_u = d_u
        else:
            self.d_u = self.bound / 5
        if d_l:
            self.d_l = d_l
        else:
            self.d_l = self.d_u / 1e5
        self.cooling_scheme = cooling_scheme
        self.max_restart = max_restart
        self.chi_f = chi_f
        self.archive_size = archive_size
        if d_min:
            self.d_min = d_min
        else:
            self.d_min = self.bound / 100
        if d_sim:
            self.d_sim = d_sim
        else:
            self.d_sim = self.d_min / 100
        # random seed for reproducibility
        np.random.seed(seed=seed)
        # generate initial solution by drawing from a uniform random variable over the feasible region
        self.x0 = np.random.uniform(-bound, bound, n)
        self.f_x0 = f(self.x0)
        # initialise archive
        self.archive_x = [self.x0]
        self.archive_f = [self.f_x0]
        # generate non-zero elements of the diagonal matrix D, represented as a (n, ) vector d
        self.d = np.ones(n)
        # initialise iteration and acceptance counters
        self.iter = 1
        self.acc = 1

    def initialise_T(self):
        """
        Method which initialises the search temperature T_0 by performing an initial search in which all increases in f are accepted. T_0 is calculated to give an average probability chi_0 of a solution that increases f.

        Returns
        ----------
        x_prev : (n, ) array
            Vector of current solution.
        f_prev : float
            Objective function evaluated at current solution.
        T_0 : float
            Initial search temperature.
        """
        # initial solution x0
        x_prev = self.x0
        # evaluate objective function at initial solution
        f_prev = self.f_x0
        # array of objective increases
        df_plus = []

        # run the initial search for specified maximum length of Markov chain
        while self.iter < self.max_chain:
            # generate a uniform rv in the range (-1,1) of equal dimension as x
            u = np.random.uniform(-1, 1, self.n)
            # find the proposal step dx
            dx = self.d * u
            # perturb the current position x_prev by the proposal step dx
            x_new = x_prev + dx
            # reject proposal if outside of feasible region
            if max(abs(x_new)) > self.bound:
                # increment iteration counter
                self.iter += 1
                continue
            # evaluate change in objective function
            f_new = self.f(x_new)
            df = f_new - f_prev
            # record increases in objective function
            if df > 0:
                df_plus.append(df)
            # update d only if objective function is decreased(?)
            r = np.abs(dx)
            self.d = (1 - self.alpha) * self.d + self.alpha * self.omega * r
            self.d = np.clip(self.d, self.d_l, self.d_u)
            # accept all feasible trial solutions
            x_prev = x_new
            f_prev = f_new
            # update archive
            self.update_archive(x_prev, f_prev)
            # increment iteration and acceptance counters
            self.iter += 1
            self.acc += 1

        # average objective function increase
        df_bar = np.mean(df_plus)
        T_0 = - df_bar / np.log(self.chi_0)
        return x_prev, f_prev, T_0

    def update_T(self, T, f_T):
        """
        Method which updates the annealing temperature according to either: the exponential CS outlined by Kirkpatrick et al. [1982]; the adaptive CS outlined by Huang et al. [1986].

        Parameters
        ----------
        T : float
            Current temperature.
        f_T : (k, ) array
            Vector of objective function acceptances at current temperature.

        Returns
        ----------
        alpha * T : float
            Decremented temperature.
        """
        if self.cooling_scheme == 'exponential':
            alpha = 0.95
        else:
            if len(f_T) > 1:
                sigma = np.std(f_T)
                alpha = max(0.5, np.exp(-0.7 * T / sigma))
            else:
                alpha = 0.95
        return alpha * T

    def update_archive(self, x_j, f_j):
        """
        Method which updates the class archives of best solutions.

        Parameters
        ----------
        x_j : (n, ) array
            A new n-dimensional candidate solution for archiving.
        f_j : float
            Objective function evaluated at the candidate solution x_j.
        """
        # worst archived solution
        g_ind = np.argmax(self.archive_f)
        f_g = self.archive_f[g_ind]

        # Euclidean distance between new and archived solutions
        distance = norm(self.archive_x - x_j, axis=1)
        # archived solution x_e which most closely resembles new solution x_j
        e_ind = np.argmin(distance)
        d_ej = distance[e_ind]
        f_e = self.archive_f[e_ind]

        # archive is not full
        if len(self.archive_x) < self.archive_size:
            # archive new solution x_j if it is sufficiently dissimilar to all the solutions archived
            if d_ej > self.d_min:
                self.archive_x.append(x_j)
                self.archive_f.append(f_j)

        # archive is full
        else:
            # archive x_j if it is sufficiently dissimilar to all the solutions archived and better than the worst
            if d_ej > self.d_min and f_j < f_g:
                # new solution x_j replaces the worst archived solution x_g
                self.archive_x.pop(g_ind)
                self.archive_f.pop(g_ind)
                self.archive_x.append(x_j)
                self.archive_f.append(f_j)
            # archive x_j if it is not sufficiently dissimilar to all the solutions archived and better than the worst
            if d_ej < self.d_min and f_j < f_g:
                # new solution x_j replaces the closest archived solution x_e
                self.archive_x.pop(e_ind)
                self.archive_f.pop(e_ind)
                self.archive_x.append(x_j)
                self.archive_f.append(f_j)
            # archive x_j if it is not the best solution so far and sufficiently similar to but better than x_e
            if d_ej < self.d_sim and f_j < f_e:
                self.archive_x.pop(e_ind)
                self.archive_f.pop(e_ind)
                self.archive_x.append(x_j)
                self.archive_f.append(f_j)

    def run(self):
        """
        Method which runs the SA algorithm.

        Returns
        -------
        x_min : (n, ) array
            Minimised solution.
        f_min : float
            Objective function evaluated at the minimised solution.
        archive_x : (k, n) array
            Matrix of k final archived solutions. Each control variable is of dimension n.
        archive_fx : (k, ) array
            Vector of objective function values evaluated at the k final archived solutions.
        """
        # perform initial search
        x_prev, f_prev, T = self.initialise_T()
        # set minimum number of acceptances for each temperature
        eta = 0.6 * self.max_chain
        # initialise iterations and acceptances for current temperature
        iter_T, acc_T = 0, 0
        # initialise restart counter
        restart_iter = 0
        # array of objective function acceptances at current temperature
        f_T = [f_prev]

        # run the search routine for a given maximum number of iterations
        while self.iter < self.max_iter:
            # generate a uniform rv in the range (-1,1) of equal dimension as x
            u = np.random.uniform(-1, 1, self.n)
            # find the proposal step dx
            dx = self.d * u
            # generate the elements of the diagonal matrix R, represented as a vector r
            r = np.abs(dx)
            # compute the actual step size associated with the proposal step dx
            step_size = np.sqrt(np.sum(r ** 2))
            # perturb the current position x_prev by the proposal step dx
            x_new = x_prev + dx
            # increment iteration counters
            self.iter += 1
            iter_T += 1
            restart_iter += 1

            # accept proposal if inside feasible region
            if max(abs(x_new)) < self.bound:
                # evaluate objective function at proposal solution
                f_new = self.f(x_new)
                # acceptance probability
                p_acc = min(1, np.exp(- (f_new - f_prev) / (T * step_size)))
                if p_acc > np.random.uniform(0, 1):
                    # reset restart counter if new best solution has been found
                    if f_new < min(self.archive_f):
                        restart_iter = 0
                    # update archive
                    self.update_archive(x_new, f_new)
                    # update non-zero elements of matrix D
                    self.d = (1 - self.alpha) * self.d + self.alpha * self.omega * r
                    self.d = np.clip(self.d, self.d_l, self.d_u)
                    # update array of objective function acceptances at current temperature
                    f_T.append(f_new)
                    # accept solution
                    x_prev = x_new
                    f_prev = f_new
                    # increment acceptance counters
                    acc_T += 1
                    self.acc += 1

            # check conditions for termination
            # terminate if no new best solution over whole Markov chain and acceptance ratio falls below threshold chi_f
            if iter_T > self.max_chain and restart_iter > self.max_chain and acc_T / iter_T < self.chi_f:
                break

            # check conditions for temperature cooling
            # decrement temperature if max Markov chain length is attained or number acceptances exceeds eta
            if iter_T > self.max_chain or acc_T > eta:
                # standard deviation of objective function values at current temperature
                T = self.update_T(T, f_T)
                # reset temperature specific counters and array
                iter_T, acc_T = 0, 0
                f_T = []

            # check conditions for restart
            if restart_iter > self.max_restart:
                # archive index of best solution
                best_ind = np.argmin(self.archive_f)
                # set current solution to best solution
                x_prev = self.archive_x[best_ind]
                f_prev = self.archive_f[best_ind]
                # reset restart counter
                restart_iter = 0

        # best minimised solution
        idx = np.argmin(self.archive_f)
        x_min = self.archive_x[idx]
        f_min = self.archive_f[idx]

        return x_min, f_min, self.archive_x, self.archive_f


def main():
    # number of dimensions and bound on feasible region
    n, bound = 5, 500
    sa = SA(f, n, bound)
    x_min, f_min, archive_x, archive_f = sa.run()


if __name__ == '__main__':
    main()

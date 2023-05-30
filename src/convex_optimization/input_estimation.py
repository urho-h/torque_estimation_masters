import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlsim
import pickle

import cvxpy as cp

from . import data_equation as de


def tikhonov_problem(meas, obsrv, gamm, regu, initial_state=None, lam=1, cmplx=False):
    '''
    This function uses the cvxpy library to solve a Tikhonov regularization problem.
    '''
    d = cp.Variable((gamm.shape[1], 1), complex=cmplx)

    if initial_state is None:
        x = cp.Variable((obsrv.shape[1], 1), complex=cmplx)
    else:
        x = initial_state

    measurements = cp.Parameter(meas.shape)
    measurements.value = meas

    objective = cp.Minimize(cp.sum_squares(measurements - obsrv @ x - gamm @ d) + lam * cp.sum_squares(regu @ d))

    prob = cp.Problem(objective)
    prob.solve()

    if initial_state is None:
        x_value = x.value
    else:
        x_value = initial_state

    return d.value, x_value


def lasso_problem(meas, obsrv, gamm, regu, initial_state=None, lam=1, cmplx=False):
    '''
    This function uses the cvxpy library to solve a LASSO problem.
    '''
    d = cp.Variable((gamm.shape[1], 1), complex=cmplx)

    if initial_state is None:
        x = cp.Variable((obsrv.shape[1], 1), complex=cmplx)
    else:
        x = initial_state

    measurements = cp.Parameter(meas.shape)
    measurements.value = meas

    objective = cp.Minimize(cp.sum_squares(measurements - obsrv @ x - gamm @ d) + lam * cp.pnorm(regu @ d, 1))

    prob = cp.Problem(objective)
    prob.solve()

    if initial_state is None:
        x_value = x.value
    else:
        x_value = initial_state

    return d.value, x_value


def elastic_net_problem(meas, obsrv, gamm, regu, initial_state=None, lam1=1, lam2=1, cmplx=False):
    '''
    This function uses the cvxpy library to solve an elastic net problem.
    '''
    d = cp.Variable((gamm.shape[1], 1), complex=cmplx)

    if initial_state is None:
        x = cp.Variable((obsrv.shape[1], 1), complex=cmplx)
    else:
        x = initial_state

    measurements = cp.Parameter(meas.shape)
    measurements.value = meas

    objective = cp.Minimize(
        cp.sum_squares(measurements - obsrv @ x - gamm @ d) + lam1 * cp.sum_squares(regu @ d) + lam2 * cp.pnorm(regu @ d, 1)
    )

    prob = cp.Problem(objective)
    prob.solve()

    if initial_state is None:
        x_value = x.value
    else:
        x_value = initial_state

    return d.value, x_value


def get_data_equation_matrices(A, B, C, D, n, bs):
    D2 = de.second_difference_matrix(bs, B.shape[1])
    O = de.O(A, C, bs)
    G = de.gamma(A, B, C, bs)
    L = np.eye(bs*B.shape[1])

    return O, G, D2, L


def progressbar(it, prefix="", size=60, out=sys.stdout):
    """
    A function used to display a progress bar in the console.
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def L_curve(sys, measurements, times, lambdas, use_zero_init=True, use_l1=False, use_trend=False):
    dt = np.mean(np.diff(times))
    bs = len(times)
    n = len(times)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_trend:
        regularization = D2
    else:
        regularization = L

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    input_estimates = []

    y = measurements.reshape(-1,1)

    for i in progressbar(range(len(lambdas)), "Calculating estimates :", len(lambdas)):
        if use_l1:
            estimate, x_init = tikhonov_problem(y, O, G, regularization, initial_state=x_init, lam=lambdas[i])
        else:
            estimate, x_init = lasso_problem(y, O, G, regularization, initial_state=x_init, lam=lambdas[i])
        input_estimates.append(estimate)

    norm, res_norm = [], []
    for i in range(len(lambdas)):
        norm.append(np.linalg.norm(y - G @ input_estimates[i]))
        res_norm.append(np.linalg.norm(D2 @ input_estimates[i]))

    return norm, res_norm


def pareto_curve(sys, load, measurements, times, lambdas, use_zero_init=True, use_trend=False):
    dt = np.mean(np.diff(times))
    bs = len(times)
    n = len(times)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_trend:
        regularization = D2
    else:
        regularization = L

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    y = measurements.reshape(-1,1)
    u = load.reshape(-1,1)

    input_estimates = []

    for i in progressbar(range(len(lambdas)), "Calculating norms :", len(lambdas)):
        intermediate = []
        for j in range(len(lambdas)):
            estimate, _ = elastic_net_problem(
                y,
                O,
                G,
                regularization,
                initial_state=x_init,
                lam1=lambdas[i],
                lam2=lambdas[j]
            )
            intermediate.append(estimate)
        input_estimates.append(intermediate)

    norms_l1, norms_l2, res_norms = [], [], []

    for i in progressbar(range(len(lambdas)), "Calculating norms :", len(lambdas)):
        norm_l1, norm_l2, res_norm = [], [], []
        for j in range(len(lambdas)):
            norm_l1.append(np.linalg.norm(y - G @ input_estimates[i][j], ord=1))
            norm_l2.append(np.linalg.norm(y - G @ input_estimates[i][j]))
            res_norm.append(np.linalg.norm(regularization @ input_estimates[i][j]))
        norms_l1.append(norm_l1)
        norms_l2.append(norm_l2)
        res_norms.append(res_norm)

    return norms_l1, norms_l2, res_norms


def data_eq_simulation(sys, times, load, bs):
    dt = np.mean(np.diff(times))
    n = len(times)

    A, B, C, D = sys
    omat = de.O(A, C, bs)
    gmat = de.gamma(A, B, C, bs)

    u = load.T.reshape(-1,1)

    return gmat @ u


def estimate_input(sys, measurements, batch_size, overlap, times, lam=0.1, lam2=0.1, use_zero_init=True, use_lasso=False, use_elastic_net=False, use_trend_filter=False, pickle_data=False, fn="input_estimates_"):
    dt = np.mean(np.diff(times))
    n = len(times)
    bs = batch_size + 2*overlap
    loop_len = int(n/batch_size)

    A, B, C, D = sys
    O, G, D2, L = get_data_equation_matrices(A, B, C, D, n, bs)

    if use_trend_filter:
        regul_matrix = D2
    else:
        regul_matrix = L

    if use_zero_init:
        x_init = np.zeros((O.shape[1], 1))
    else:
        x_init = None

    input_estimates = []

    # for initial state estimation
    C_full = np.eye(B.shape[0])
    omat = de.O(A, C_full, bs)
    gmat = de.gamma(A, B, C_full, bs)

    for i in progressbar(range(loop_len), "Calculating estimates: ", loop_len):
        if i == 0:
            batch = measurements[:bs,:]
        elif i == loop_len-1:
            batch = np.zeros((bs, measurements.shape[1]))
            # zero padding to finish estimation loop correctly
        else:
            batch = measurements[i*batch_size-overlap:(i+1)*batch_size+overlap,:]

        y = batch.reshape(-1,1)

        if use_lasso:
            estimate, x_init = lasso_problem(y, O, G, regul_matrix, initial_state=x_init, lam=lam)
        elif use_elastic_net:
            estimate, x_init = elastic_net_problem(y, O, G, regul_matrix, initial_state=x_init, lam1=lam, lam2=lam2)
        else:
            estimate, x_init = tikhonov_problem(y, O, G, regul_matrix, initial_state=x_init, lam=lam)

        x_est = omat @ x_init + gmat @ estimate

        x_init = x_est[-A.shape[0]:,:]

        input_estimates.append(estimate)

        if pickle_data:
            with open(fn + str(i) + ".pickle", 'wb') as handle:
                pickle.dump([estimate, x_est], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return input_estimates
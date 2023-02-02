import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import opentorsion as ot
import pickle

import sys
sys.path.append('../') # temporarily adds '../' to pythonpath so the drivetrain module is imported

import drivetrain
import handle_data

def O(A, C, n):
    '''
    Create the extended observability matrix used in the data equation.

    Parameters:

    A : numpy.ndarray
        The state matrix of the state-space system
    C : numpy.ndarray
        The observation matrix of the state-space system
    n : float
        number of measurements

    Returns:

    O : numpy.ndarray, shape(n, number of state variables)
        The extended observability matrix
    '''
    O = C
    for k in range(1, n):
        O = np.vstack((O, C @ np.linalg.matrix_power(A, k)))

    return O

def gamma(A, B, C, n):
    '''
    Create the impulse response matrix used in the data equation.

    Parameters:

    A : numpy.ndarray
        The state matrix of the state-space system
    B : numpy.ndarray
        The input matrix of the state-space system
    C : numpy.ndarray
        The observation matrix of the state-space system
    n : float
        number of measurements

    Returns:

    gamma : numpy.ndarray, shape(n*number of state variables, n*number of state variables)
        The impulse response matrix
    '''
    # first column
    gamma_column_first = np.zeros(B.shape)
    for k in range(1, n):
        gamma_column_first = np.vstack((gamma_column_first, C @ np.linalg.matrix_power(A, k) @ B))

    # build complete matrix
    gamma = gamma_column_first
    current_col = 1
    for s in range(1, n):
        gamma_rows = np.zeros(B.shape)
        current_row = current_col
        for k in range(1, n):
            if current_row > 0:
                row_val = np.zeros(B.shape)
                current_row -= 1
            else:
                row_val = C @ np.linalg.matrix_power(A, k-current_col) @ B
            gamma_rows = np.vstack((gamma_rows, row_val))

        gamma = np.hstack((gamma, gamma_rows))
        current_col += 1

    return gamma

def L(input_shape):
    '''
    The regularization matrix L. Currently an identity matrix of length n.

    Parameters:

    meas_shape : float
        Input vector shape used to determine L matrix shape.

    Returns:

    L : ndarray
        The regularization matrix
    '''
    L = np.eye(input_shape)

    return L

def convex_optimization_problem(meas, O, gamma, lam=100, L=None):
    '''
    Convex optimization methods using the cvxpy library.
    '''
    d = cp.Variable((gamma.shape[1], 1), complex=False)
    x = np.array(meas[:6,0]).reshape(6, 1) # first measurement used as the initial state
    print(d.shape)
    print(gamma.shape)
    objective = cp.Minimize(cp.sum_squares(meas - O @ x - gamma @ d) + lam * cp.pnorm(L @ d, 1))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.CVXOPT)

    return prob.value

def test_data_equation_matrices():
    '''
    An arbitrary test to inspect matrix shapes.
    '''
    A = np.array([[1.5, 1], [-0.7, 0]])
    B = np.array([[1, 0.3], [0.5, 1]])
    C = np.array([[0.1, 0], [0, 0.2]])

    n = 100

    o = O(A, C, n)

    gamm = gamma(A, B, C, n)

    return o.shape, gamm.shape

def sirm_experiment():
    '''
    A test with the mechanical system used in the SIRM conference paper.
    '''
    pathname = "../../data/rpm1480.0.pickle"
    time, theta, omega, motor, load = handle_data.get_sirm_dataset(pathname)
    n = 100 # number of measurements
    t_start = 600000 # timestep to start at
    dof = 3 # the example system DOF = 3
    meas, input = handle_data.construct_measurement(theta, omega, motor, load, dof, n, t_start)

    assembly = drivetrain.drivetrain_3dof()
    A, B = drivetrain.state_matrices(assembly)
    C = np.eye(B.shape[0])

    observe = O(A, C, n)
    impulse = gamma(A, B, C, n)
    regularize = L(impulse.shape[1])

    estimate = convex_optimization_problem(meas, observe, impulse, L=regularize)
    plt.plot(estimate)
    plt.show()

if __name__ == "__main__":
    sirm_experiment()

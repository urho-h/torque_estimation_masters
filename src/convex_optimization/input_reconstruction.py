import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import opentorsion as ot
import pickle

def get_dataset():
    '''
    Unpickles a dataset containing the shaft torque and
    excitation data of a simulated electric drive, with
    open-loop control and 1480 RPM operating speed reference.
    The data was used in the SIRM conference paper.

    Returns:

    time : list
        simulation time in seconds
    thetas : list
        measured rotational values at each drivetrain node (rad)
    omegas : list
        measured mechanical rotating speed at each drivetrain node (rad/s)
    '''

    pathname = "../data/rpm1480.0.pickle"
    try:
        with open(pathname, 'rb') as handle:
            dataset = pickle.load(handle)
            time, theta, omega = dataset[0], dataset[1], dataset[2]
    except EOFError:
        print("corrupted dataset")

    theta = np.array(theta)
    omega = np.array(omega)

    return time, theta, omega

def construct_measurement(time, theta, omega, n, t_start):
    '''
    Builds the batch measurement matrix.
    '''
    measurements = np.vstack((theta[0+t_start,:].reshape(3,1), omega[0+t_start,:].reshape(3,1)))

    for i in range(1+t_start, n+t_start):
        measurements = np.vstack((measurements, np.vstack((theta[i,:].reshape(3,1), omega[i,:].reshape(3,1)))))

    return measurements

def drivetrain_3dof():
    '''
    Mechanical drivetrain as an openTorsion assembly instance.

    Returns:

    assembly : opentorsion assembly instance
        A 3-DOF mechanical drivetrain modelled as lumped masses and flexible shafts
        (lumped mass - shaft - lumped mass - shaft - lumped mass).
    '''
    J1 = 0.8 # disk 1 inertia
    J2 = 0.5 # disk 2 inertia
    J3 = 0.7 # disk 3 inertia
    k1 = 1.5e4 # shaft 1 stiffness
    k2 = 1e4 # shaft 2 stiffness

    disks, shafts = [], []
    shafts.append(ot.Shaft(0,1, None, None, k=k1, I=0))
    shafts.append(ot.Shaft(1,2, None, None, k=k2, I=0))
    disks.append(ot.Disk(0, I=J1))
    disks.append(ot.Disk(1, I=J2))
    disks.append(ot.Disk(2, I=J3))
    assembly = ot.Assembly(shafts, disk_elements=disks)
    _, f, _ = assembly.modal_analysis()

    return assembly

def state_matrices(assembly):
    '''
    Create state-space matrices A and B

    Returns:

    A : numpy.ndarray
        The state matrix
    B : numpy.ndarray
        The input matrix
    '''
    M, K = assembly.M(), assembly.K() # mass and sitffness matrices
    C = assembly.C_modal(M, K) # modal damping matrix, modal damping coefficient 0.02 used

    A, B = assembly.state_matrix(C=C)

    return A, B

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

    O : numpy.ndarray
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

    gamma : numpy.ndarray
        The impulse response matrix
    '''
    # first column
    gamma_column_first = np.zeros(C.shape)
    for k in range(1, n):
        gamma_column_first = np.vstack((gamma_column_first, C @ np.linalg.matrix_power(A, k) @ B))

    # build complete matrix
    gamma = gamma_column_first
    current_col = 1
    for s in range(1, n):
        gamma_rows = np.zeros(C.shape)
        current_row = current_col
        for k in range(1, n):
            if current_row > 0:
                row_val = np.zeros(C.shape)
                current_row -= 1
            else:
                row_val = C @ np.linalg.matrix_power(A, k-current_col) @ B
            gamma_rows = np.vstack((gamma_rows, row_val))

        gamma = np.hstack((gamma, gamma_rows))
        current_col += 1

    return gamma

def L(meas, n):
    '''
    The regularization matrix L. Assuming the input signal has a sparse Fourier-series representation.
    '''
    L = []
    for i in range(meas.shape[0]):
        L.append(np.exp(1j*2*np.pi*0*i/n))

    for j in range(1, meas.shape[0]):
        coeffs = []
        for i in range(meas.shape[0]):
            coeffs.append(np.exp(1j*2*np.pi*j*i/n))

        L = np.vstack((L, coeffs))

    return L

def convex_optimization_problem(meas, O, gamma, lam=100, L=None):
    '''
    Input reconstruction with convex optimization methods using the cvxpy library.
    '''
    d = cp.Variable(meas.shape, complex=True)
    x = np.array(meas[:6,0]).reshape(6, 1) # first measurement used as the initial state
    objective = cp.Minimize(cp.sum_squares(meas - O @ x - gamma @ d) + lam * cp.pnorm(L @ d, 1))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.CVXOPT)

    return prob.value

def test_data_equation_matrices():
    '''
    Manngård 2017
    '''
    A = np.array([[1.5, 1], [-0.7, 0]])
    B = np.array([[1, 0.3], [0.5, 1]])
    C = np.array([[0.1, 0], [0, 0.2]])

    n = 100

    o = O(A, C, n)

    gamm = gamma(A, B, C, n)

    return

def sirm_experiment():
    time, theta, omega = get_dataset()
    n = 100 # number of measurements
    t_start = 600000 # timestep to start at
    meas = construct_measurement(time, theta, omega, n, t_start)

    assembly = drivetrain_3dof()
    A, B = state_matrices(assembly)
    C = np.eye(B.shape[0])

    observe = O(A, C, n)
    impulse = gamma(A, B, C, n)
    regularize = L(meas, n)

    estimate = convex_optimization_problem(meas, observe, impulse, L=regularize)
    print(estimate)
    plt.plot(estimate)
    plt.show()

if __name__ == "__main__":
    sirm_experiment()

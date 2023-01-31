import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim
import pickle

import opentorsion as ot

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

    pathname = "../../data/rpm1480.0.pickle"
    try:
        with open(pathname, 'rb') as handle:
            dataset = pickle.load(handle)
            time, theta, omega, motor, load = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4]
    except EOFError:
        print("corrupted dataset")

    theta = np.array(theta)
    omega = np.array(omega)
    motor = np.array(motor)
    load = np.array(load)

    return time, theta, omega, motor, load

def construct_measurement(theta, omega, motor, load, n, t_start):
    '''
    Builds the batch measurement matrix.

    Parameters:

    theta : ndarray, shape (i,)
        rotational angle measurement
    omega : ndarray, shape (i,)
        rotational speed measurement
    motor : ndarray, shape (i,)
        motor load
    load : ndarray, shape (i,)
        external load

    Returns:

    measurements : ndarray
        angles and speeds stacked
    load : ndarray
        input matrix
    '''
    measurements = np.vstack((theta[0+t_start,:].reshape(3,1), omega[0+t_start,:].reshape(3,1)))

    for i in range(1+t_start, n+t_start):
        measurements = np.hstack((measurements, np.vstack((theta[i,:].reshape(3,1), omega[i,:].reshape(3,1)))))

    inputs = np.zeros((measurements.shape[0], n))
    # inputs[0,:] = motor[t_start:n+t_start]
    inputs[-1,:] = load[t_start:n+t_start]

    return measurements, inputs

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

def kalman_filter(A, B, C, R, Q):
    '''
    Asymptotic Kalman filter.
    The covariance matrix is computed by solving a discrete Ricatti equation.
    '''
    P = LA.solve_discrete_are(A, C, Q, R) # ricatti_equation
    # K = P @ C.T @ LA.inv(R + C @ P @ C.T) # kalman gain
    K = A @ P @ C.T @ LA.inv(R + C @ P @ C.T) # kalman gain

    # KF = dlti((np.eye(P.shape[0]) - K @ C) @ A, K, C, np.zeros(C.shape), dt=1)
    # KF = dlti(A - K @ C, K, C, np.zeros(C.shape), dt=1)

    return K

def estimation_loop(A, B, C, K, meas, load, initial_state):
    current_x = initial_state
    estimate = ((A - K @ C) @ current_x).T + B @ load[:,0] + K @ (meas[:,0])
    for i in range(1, load.shape[1]):
        new_estimate = ((A - K @ C) @ current_x).T + B @ load[:,i] + K @ (meas[:,i])
        estimate = np.vstack((estimate, new_estimate))
        current_x = estimate[-1,:]

    return estimate

if __name__ == "__main__":
    time, theta, omega, motor, load = get_dataset()
    # meas, load = construct_measurement(theta, omega, motor, load, 40000, 180000)
    meas, load = construct_measurement(theta, omega, motor, load, 40000, 580000)

    propeller_angles, propeller_speed = np.copy(meas[2,:]), np.copy(meas[5,:])
    initial_state = np.copy(meas[:,0])

    # save only measurements from node 1
    meas[1,:] *= 0
    meas[2,:] *= 0
    meas[4,:] *= 0
    meas[5,:] *= 0

    assembly = drivetrain_3dof()

    A, B = state_matrices(assembly)
    C = np.eye(A.shape[0])

    R = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]) # measurement covariance R = E{v*v'}
    Q = 1e-2*np.eye(A.shape[0]) # process covariance

    K = kalman_filter(A, B, C, R, Q)
    estimate = estimation_loop(A, B, C, K, meas, load, initial_state)

    ## plots speed at node 3
    plt.plot(propeller_speed, label="measured speed")
    plt.plot(estimate[:,5], label="estimated speed", alpha=0.5)
    plt.legend()
    plt.show()

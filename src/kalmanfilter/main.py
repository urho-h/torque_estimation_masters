import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim
import opentorsion as ot
import pickle

import sys
sys.path.append('../') # temporarily adds '../' to pythonpath so the drivetrain module is imported

import drivetrain
import handle_data

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
    time, theta, omega, motor, load = handle_data.get_dataset()
    # meas, load = construct_measurement(theta, omega, motor, load, 40000, 180000)
    meas, load = handle_data.construct_measurement(theta, omega, motor, load, 3, 40000, 580000)

    propeller_angles, propeller_speed = np.copy(meas[2,:]), np.copy(meas[5,:])
    initial_state = np.copy(meas[:,0])

    # save only measurements from node 1
    meas[1,:] *= 0
    meas[2,:] *= 0
    meas[4,:] *= 0
    meas[5,:] *= 0

    assembly = drivetrain.drivetrain_3dof()

    A, B = drivetrain.state_matrices(assembly)
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

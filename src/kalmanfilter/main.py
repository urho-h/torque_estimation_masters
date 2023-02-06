import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim, butter, lfilter
import opentorsion as ot
import pickle

import sys
sys.path.append('../') # temporarily adds '../' to pythonpath so the drivetrain module is imported

import drivetrain
import handle_data

def kalman_filter(A, B, C, R, Q, dt=1):
    '''
    Asymptotic Kalman filter as scipy dlti instance.
    The estimate covariance matrix is computed by solving a discrete Ricatti equation.
    '''
    P = LA.solve_discrete_are(A, C, Q, R) # ricatti_equation
    K = P @ C.T @ LA.inv(R + C @ P @ C.T) # kalman gain

    KF = dlti((np.eye(P.shape[0]) - K @ C) @ A, K, C, np.zeros(C.shape), dt=dt)

    return KF

def kalman_gain(A, B, C, R, Q):
    '''
    Asymptotic Kalman filter.
    The estimate covariance matrix is computed by solving a discrete Ricatti equation.
    '''
    P = LA.solve_discrete_are(A, C.T, Q, R) # ricatti_equation
    K = A @ P @ C.T @ LA.inv(R + C @ P @ C.T) # kalman gain

    return K

def estimation_loop(A, B, C, K, meas, load, initial_state):
    '''
    Update predicted states using the asymptotic KF.
    '''
    current_x = initial_state
    estimate = ((A - K @ C) @ current_x).T + B @ load[:,0] + K @ (meas[:,0])
    for i in range(1, load.shape[1]):
        new_estimate = ((A - K @ C) @ current_x).T + B @ load[:,i] + K @ (meas[:,i])
        estimate = np.vstack((estimate, new_estimate))
        current_x = estimate[-1,:]

    return estimate

def conventional_kalman_filter(A, B, C, Q, R, Y, load, m0, P0):
    '''
    Conventional Kalman filter.
    '''
    T = Y.shape[1]
    nx = m0.shape[0] # dimension of x
    x = np.zeros((nx, T))
    P = []
    x[:,0] = m0
    P.append(P0)
    I = np.eye(P0.shape[0])
    for n in range(T-1):
        # Prediction
        x_ = A @ x[:,n] + B @ load[:,n]
        P_ = A @ P[n] @ A.T + Q
        # Update
        S = C @ P_ @ C.T + R
        K = P_ @ C.T @ LA.inv(S)
        # P.append(P_ - K @ S @ K.T)
        # P.append((I - K @ C) @ P_)
        P.append((I - K @ C) @ P_ @ (I - K @ C).T + K @ R @ K.T)
        x[:,n+1] = x_ + K @ (Y[:,n] - C @ x_)

    return x, P

def low_pass_filter(signal, cutoff, fs):
    '''
    A fifth-order Butterworth low-pass filter.
    '''
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(5, normalized_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal

if __name__ == "__main__":
    '''
    Currently works only for a 3-DOF system.
    '''

    # load dataset
    pathname = '../../data/drivetrain_simulation.pickle' # simulated measurements
    with open(pathname, 'rb') as handle:
        dataset = pickle.load(handle)
        t, U, tout, yout = dataset[0], dataset[1], dataset[2], dataset[3] # tout is measurement timesteps
        # yout contains angle and speed measurements
        # first three rows are rotational values at each node, last three rotational speeds
        # each column represents a timestep

    # get state-space matrices created using openTorsion
    assembly = drivetrain.drivetrain_3dof()
    A, B = drivetrain.state_matrices(assembly)
    C = np.eye(A.shape[0])

    R = 1e-4*np.eye(C.shape[0]) # measurement covariance, shape (n_sensors, n_sensors)
    Q = np.diag(np.array([0, 0, 0, 1, 1, 1])) # process (speed) covariance, shape (n_states, n_states)

    ### add gaussian white noise to the measurement (measurement and process noise) ###
    r = np.random.multivariate_normal(np.zeros(R.shape[0]), R, tout.shape[0])
    q = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, tout.shape[0])

    yout_noise = (yout + r + q).T # shape (n_states, n_timesteps)
    yout_lowpass = low_pass_filter(yout_noise, 30, 1/np.mean(np.diff(t))) # low-pass filter the measurement

    meas = np.vstack([yout_lowpass[1,:], yout_lowpass[4,:]]) # use measurements at node 2

    # C and R matrices redefined
    C = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
    R = 1e-4*np.eye(C.shape[0])

    ####### Conventional Kalman filter #######
    m0 = np.copy(yout[0,:]) # initial state guess
    P0 = 1e-0*np.eye(A.shape[0]) # randomly chosen estimate covariance
    x_kalman, cov_kalman = conventional_kalman_filter(A, B, C, Q, R, meas, U, m0, P0)

    ## plots speed at node 1
    plt.figure()
    plt.plot(x_kalman[-3,:], label="estimated speed", linestyle='--', color='green')
    plt.plot(yout_lowpass[-3,:], label="measured speed with noise", alpha=0.8, color='b')
    plt.plot(yout[:,-3], label="measured speed (no noise)", color='r')
    plt.ylim(-5, 60)
    plt.title("node 1")
    plt.legend()

    ## plots speed at node 2
    plt.figure()
    plt.plot(x_kalman[-2,:], label="estimated speed", linestyle='--', color='green')
    plt.plot(yout_lowpass[-2,:], label="measured speed with noise", alpha=0.8, color='b')
    plt.plot(yout[:,-2], label="measured speed (no noise)", color='r')
    plt.ylim(-5, 60)
    plt.title("node 2")
    plt.legend()

    ## plots speed at node 3
    plt.figure()
    plt.plot(x_kalman[-1,:], label="estimated speed", linestyle='--', color='green')
    plt.plot(yout_lowpass[-1,:], label="measured speed with noise", alpha=0.8, color='b')
    plt.plot(yout[:,-1], label="measured speed (no noise)", color='r')
    plt.ylim(-5, 60)
    plt.legend()
    plt.title("node 3")
    plt.show()

# def sirm_experiment():
#     pathname = "../../data/rpm1480.0.pickle"
#     time, theta, omega, motor, load = handle_data.get_sirm_dataset(pathname)
#     meas, load = handle_data.construct_measurement(theta, omega, motor, load, 3, 40000, 180000, KF=True)
#     # meas, load = handle_data.construct_measurement(theta, omega, motor, load, 3, 40000, 580000, KF=True)

#     propeller_angles, propeller_speed = np.copy(meas[2,:]), np.copy(meas[5,:])
#     initial_state = np.copy(meas[:,0])

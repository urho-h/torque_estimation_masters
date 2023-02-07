import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim, butter, lfilter
from scipy.integrate import solve_ivp
import opentorsion as ot
import pickle

import sys
sys.path.append('../') # temporarily adds '../' to pythonpath so the drivetrain module is imported

import drivetrain
import handle_data

class PredictionDerivative:
    '''
    This class is used in the calculation of predicted state in the conventional Kalman filter algorithm.
    '''
    def __init__(self, A, B, u0):
        self.A = A
        self.B = B
        self.u = u0

    def update_load(self, u):
        self.u = u

    def f(self, t, x):
        '''State derivative function used by scipy.integrate.solve_ivp (see. conventional_kalman_filter function)'''

        return self.A @ x + self.B @ self.u

def conventional_kalman_filter(A, B, C, Q, R, time, Y, load, m0, P0):
    '''
    Conventional Kalman filter.
    '''
    T = Y.shape[1]
    nx = m0.shape[0] # dimension of x
    x = np.zeros((nx, T))
    x[:,0] = m0
    P = P0
    I = np.eye(P0.shape[0])
    predd = PredictionDerivative(A, B, load[:,0])
    for n in range(T-1):
        # Prediction
        dt = time[n+1]-time[n]
        x_hat = x[:,n]
        u_hat = load[:,n]
        predd.update_load(u_hat)
        sol = solve_ivp(predd.f, (time[n], time[n]+dt), x_hat)
        x_ = sol.y[:,-1]
        P_ = A @ P @ A.T + Q
        # Update
        S = C @ P_ @ C.T + R
        K = P_ @ C.T @ LA.inv(S)
        P = (I - K @ C) @ P_
        # P = (I - K @ C) @ P_ @ (I - K @ C).T + K @ R @ K.T
        y = Y[:,n]
        x[:,n+1] = x_ + K @ (y - C @ x_)

    return x

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
    State estimation in a 3-DOF drivetrain. The speed at the end of the drivetrain (node 3) is estimated with measurements from nodes 2 and 3 and known input. Currently works only for a 3-DOF system.
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

    R = 1e-5*np.eye(C.shape[0]) # measurement covariance, shape (n_sensors, n_sensors)
    Q = 1e-3*np.diag(np.array([0, 0, 0, 1, 1, 1])) # process (speed) covariance, shape (n_states, n_states)

    ### add gaussian white noise to the measurement (measurement and process noise) ###
    r = np.random.multivariate_normal(np.zeros(R.shape[0]), R, tout.shape[0])
    q = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, tout.shape[0])

    yout_noise = (yout + r + q).T # shape (n_states, n_timesteps)
    yout_lowpass = low_pass_filter(yout_noise, 20, 1/np.mean(np.diff(t))) # low-pass filter the measurement

    meas = np.vstack([yout_lowpass[0,:], yout_lowpass[1,:], yout_lowpass[3,:], yout_lowpass[4,:]]) # use speed measurements at node 1 and 3

    # C and R matrices redefined
    C = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
    R = 1e-4*np.eye(C.shape[0])
    Q = 1e-2*np.diag(np.array([0, 0, 0, 1, 1, 1])) # process (speed) covariance, shape (n_states, n_states)

    ####### Conventional Kalman filter #######
    m0 = np.copy(yout[0,:]) # initial state guess
    P0 = 1e-3*np.eye(A.shape[0]) # randomly chosen estimate covariance
    x_kalman = conventional_kalman_filter(A, B, C, Q, R, t, meas, U, m0, P0)

    ## plot speed at nodes
    for i in range(assembly.dofs):
        plt.figure()
        plt.plot(x_kalman[i+3,:], label="estimated speed", linestyle='--', color='green')
        plt.plot(yout_lowpass[i+3,:], label="measured speed with noise", alpha=0.8, color='b')
        plt.plot(yout[:,i+3], label="measured speed (no noise)", color='r')
        plt.ylim(-5, 60)
        plt.title("node " + str(i + 1))
        plt.legend()

    plt.show()

# def sirm_experiment():
#     pathname = "../../data/rpm1480.0.pickle"
#     time, theta, omega, motor, load = handle_data.get_sirm_dataset(pathname)
#     meas, load = handle_data.construct_measurement(theta, omega, motor, load, 3, 40000, 180000, KF=True)
#     # meas, load = handle_data.construct_measurement(theta, omega, motor, load, 3, 40000, 580000, KF=True)

#     propeller_angles, propeller_speed = np.copy(meas[2,:]), np.copy(meas[5,:])
#     initial_state = np.copy(meas[:,0])

# def kalman_filter(A, B, C, R, Q, dt=1):
#     '''
#     Asymptotic Kalman filter as scipy dlti instance.
#     The estimate covariance matrix is computed by solving a discrete Ricatti equation.
#     '''
#     P = LA.solve_discrete_are(A, C, Q, R) # ricatti_equation
#     K = P @ C.T @ LA.inv(R + C @ P @ C.T) # kalman gain

#     KF = dlti((np.eye(P.shape[0]) - K @ C) @ A, K, C, np.zeros(C.shape), dt=dt)

#     return KF

# def kalman_gain(A, B, C, R, Q):
#     '''
#     Asymptotic Kalman filter.
#     The estimate covariance matrix is computed by solving a discrete Ricatti equation.
#     '''
#     P = LA.solve_discrete_are(A, C.T, Q, R) # ricatti_equation
#     K = A @ P @ C.T @ LA.inv(R + C @ P @ C.T) # kalman gain

#     return K

# def estimation_loop(A, B, C, K, meas, load, initial_state):
#     '''
#     Update predicted states using the asymptotic KF.
#     '''
#     current_x = initial_state
#     estimate = ((A - K @ C) @ current_x).T + B @ load[:,0] + K @ (meas[:,0])
#     for i in range(1, load.shape[1]):
#         new_estimate = ((A - K @ C) @ current_x).T + B @ load[:,i] + K @ (meas[:,i])
#         estimate = np.vstack((estimate, new_estimate))
#         current_x = estimate[-1,:]

#     return estimate

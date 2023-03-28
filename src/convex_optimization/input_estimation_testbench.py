import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import lsim, dlsim, butter, lfilter
import pickle

from tqdm import tqdm
import opentorsion as ot
import cvxpy as cp

import data_equation as de

import sys
sys.path.append('../') # temporarily adds '../' to pythonpath so the drivetrain module is imported

import testbench_MSSP as tb


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


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    c_mod = np.insert(C, 0, np.zeros((1, C.shape[1])), 0)
    c_mod[0,0] += 1 # additional measurement: motor speed
    C_mod1 = np.insert(c_mod, 3, np.zeros((1, c_mod.shape[1])), 0)
    C_mod1[3,22] += 1.9e5 # additional measurement: first shaft torque, assumed to be equal with motor torque
    C_mod = np.insert(C_mod1, 5, np.zeros((1, C_mod1.shape[1])), 0)
    C_mod[5,22+19] += 2.0e4 # additional measurement: second torque transducer

    return A, B, C_mod, D


def get_data_equation_matrices(A, B, C, D, n, bs):
    D2 = de.second_difference_matrix(bs, B.shape[1])
    O = de.O(A, C, bs)
    G = de.gamma(A, B, C, bs)

    return O, G, D2


def estimate_input(measurements, bs, n, dt):
    loop_len = int(n/bs)

    A, B, C, D = get_testbench_state_space(dt)
    O, G, D2 = get_data_equation_matrices(A, B, C, D, n, bs)

    x_tikhonov = np.zeros((O.shape[1], 1))
    x_lasso = np.zeros((O.shape[1], 1))
    tikh_estimates = []
    lasso_estimates = []

    # for initial state estimation
    C_full = np.eye(B.shape[0])
    omat = de.O(A, C_full, bs)
    gmat = de.gamma(A, B, C_full, bs)

    for i in tqdm(range(loop_len)):
        batch = measurements[i*bs:(i+1)*bs,:]
        y_noise = batch.reshape(-1,1)

        tikhonov_estimate, x_tikhonov = tikhonov_problem(y_noise, O, G, D2, initial_state=x_tikhonov, lam=0.05)
        lasso_estimate, x_lasso = lasso_problem(y_noise, O, G, D2, initial_state=x_lasso, lam=0.05)

        x_est_t = omat @ x_tikhonov + gmat @ tikhonov_estimate
        x_tikhonov = x_est_t[-A.shape[0]:,:]

        x_est_l = omat @ x_lasso + gmat @ lasso_estimate
        x_lasso = x_est_l[-A.shape[0]:,:]

        tikh_estimates.append(tikhonov_estimate)
        lasso_estimates.append(lasso_estimate)

    return tikh_estimates, lasso_estimates


def plot_estimates(t_motor, t_sensor, tau_motor, tau_propeller, tikh_estimates, lasso_estimates, loop_len, interval):
    t_start, t_end = interval
    plt.figure()
    for i in range(loop_len):
        plt.plot(t_motor[i*bs:(i+1)*bs], tau_motor[t_start+i*bs:t_start+(i+1)*bs], linestyle='solid', color='black')
        plt.plot(t_sensor[3*i*bs:3*(i+1)*bs:3], tikh_estimates[i][::2], linestyle='dotted', alpha=0.2, color='red')
        plt.plot(t_sensor[3*i*bs:3*(i+1)*bs:3], lasso_estimates[i][::2], linestyle='dashed', color='blue')

    plt.legend(('known input', 'Tikhonov estimate', 'LASSO estimate'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Motor side input estimates (H-P trend filtering)')

    plt.figure()
    for i in range(loop_len):
        plt.plot(t_motor[i*bs:(i+1)*bs], tau_propeller[t_start+i*bs:t_start+(i+1)*bs], linestyle='solid', color='black')
        plt.plot(t_sensor[3*i*bs:3*(i+1)*bs:3], tikh_estimates[i][1::2], linestyle='dotted', alpha=0.5, color='red')
        plt.plot(t_sensor[3*i*bs:3*(i+1)*bs:3], lasso_estimates[i][1::2], linestyle='dashed', color='blue')

    plt.legend(('known input', 'Tikhonov estimate', 'LASSO estimate'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Propeller side input estimates (H-P trend filtering)')

    plt.show()


def time_domain_test(time, motor, propeller):
    dt = np.mean(np.diff(time))
    A, B, C, D = get_testbench_state_space(dt)

    U = np.vstack((motor, -propeller)).T

    tout, yout, xout = dlsim((A, B, C, D, dt), t=time, u=U)

    return tout, yout


def plot_time_domain_test(t_sensor, omega1, torq1, tout, yout):
    plt.figure()
    plt.plot(t_sensor, torq1, label='Torque sensor 1')
    plt.plot(tout, yout[:,-1], label='Simulated torque sensor 1 (MSSP)')
    plt.legend()

    plt.figure()
    plt.plot(t_sensor, omega1, label='Speed sensor 1')
    plt.plot(tout, yout[:,0], label='Simulated speed sensor 1 (MSSP)')
    plt.legend()
    plt.show()

    plt.show()


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
    # time | MotorTorqueSet | MotorTorque | MotorVelocitySet | MotorVelocity | PropellerTorqueSet | PropellerTorque | PropellerVelocitySet | PropellerVelocity
    m_data = np.genfromtxt('../../data/step_test_long/step_test_motor.csv', delimiter=',')
    motor_data = np.delete(m_data, 0, 0) # delete header row

    t_motor = motor_data[:,0]
    tau_motor = low_pass_filter(motor_data[:,2], 400, 1000)
    omega_motor = low_pass_filter(motor_data[:,4]*(2*np.pi/60), 400, 1000)
    tau_propeller = low_pass_filter(motor_data[:,6], 400, 1000)

    dt = np.mean(np.diff(t_motor))
    n = 5000
    bs = 500
    t_start = 25000 # milliseconds
    t_end = 30000 # milliseconds

    # time | enc1_ang | enc1_time | enc2_ang | enc2_time | enc3_ang | enc3_time | enc4_ang | enc4_time | enc5_ang | enc5_time | acc1 | acc2 | acc3 | acc4 | Torq1 | Torq2
    s_data = np.genfromtxt('../../data/step_test_long/step_test_sensor.csv', delimiter=',')
    sensor_data = np.delete(s_data, 0, 0) # delete header row

    t_sensor = sensor_data[:,0]
    enc1 = sensor_data[:,1]*(np.pi/180)
    enc2 = sensor_data[:,3]*(np.pi/180)

    omega1 = low_pass_filter(np.gradient(enc1, t_sensor), 400, 3000)
    omega2 = low_pass_filter(np.gradient(enc2, t_sensor), 400, 3000)

    torq1 = low_pass_filter(sensor_data[:,-2]*1/10, 400, 3000)
    torq2 = low_pass_filter(sensor_data[:,-1]*1/4, 400, 3000)

    ##### Simulate with measured motor and propeller torques #####
    # tout, yout = time_domain_test(t_motor, tau_motor, tau_propeller)
    # plot_time_domain_test(t_sensor, omega1, torq1, tout, yout)

    filename = "estimates/estimates_l0_01_lowpass_motor_known.pickle"
    use_save_data = False
    pickle_results = True

    if use_save_data:
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)
            tikh, lasso = dataset[0], dataset[1]

    else:
        measurements_noise = np.vstack((
            omega_motor[t_start:t_end],
            omega1[3*t_start:3*t_end:3],
            omega2[3*t_start:3*t_end:3],
            tau_motor[t_start:t_end],
            torq1[3*t_start:3*t_end:3],
            torq2[3*t_start:3*t_end:3],
        )).T

        tikh, lasso = estimate_input(measurements_noise, bs, n, dt)

        if pickle_results:
            with open(filename, 'wb') as handle:
                pickle.dump([tikh, lasso], handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_estimates(t_motor, t_sensor, tau_motor, tau_propeller, tikh, lasso, int(n/bs), (t_start, t_end))

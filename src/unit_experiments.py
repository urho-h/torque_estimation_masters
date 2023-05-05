import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import lsim, dlsim, butter, lfilter
import pickle
from nptdms import TdmsFile

import testbench_MSSP as tb
from convex_optimization import input_estimation as ie
from kalmanfilter import kalmanfilter as kf


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 12,
    "figure.figsize": (6,4),
})


def read_motor_tdms(fn, start, stop):
    with TdmsFile.open(fn) as tdms_file:
        group = tdms_file['Measurements']
        time = group['Time'][start:-stop]
        time = (time-time[0])*25e-9

        all_channels = group.channels()
        print(all_channels[1][:].shape)

        motor_speed = group['MotorVelocity'][start:-stop]
        motor_setpoint = group['MotorTorqueSet'][start:-stop]
        motor = group['MotorTorque'][:]
        propeller_setpoint = group['PropellerTorqueSet'][start:-stop]
        propeller = group['PropellerTorque'][:]#[start:-stop]

    return time, motor_setpoint, motor, propeller_setpoint, propeller


def read_sensor_tdms(fn, start, stop):
    with TdmsFile.open(fn) as tdms_file:
        group = tdms_file['Measurements']

        time = group['time'][start:-stop]
        time = (time-time[0])*25e-9

        en1time = group['en1time'][start:-stop]
        en1time = (en1time-en1time[0])*25e-9
        en1angle = (group['en1angle'][start:-stop])*(2*np.pi/20000)

        en2time = group['en2time'][start:-stop]
        en2time = (en2time-en2time[0])*25e-9
        en2angle = (group['en2angle'][start:-stop])*(2*np.pi/20000)

        speed1 = np.gradient(en1angle, en1time)
        speed2 = np.gradient(en2angle, en2time)

        torque1 = (group['torque1'][start:-stop])/10
        torque2 = (group['torque2'][start:-stop])/4

    return time, speed1, speed2, torque1, torque2


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def plot_L_curve(times, dt, load, show_plot=False, pickle_data=False):
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    tout, yout, _ = dlsim((A, B, C, D, dt), u=load, t=times)
    e3 = np.random.normal(0, .01, yout.shape)
    y_noise = yout + e3

    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 10, 100]

    l_norm, residual_norm = ie.L_curve(sys, y_noise, tout, lambdas)

    if show_plot:
        plt.yscale("log")
        plt.xscale("log")
        plt.scatter(l_norm, residual_norm, color='blue')
        plt.xlabel("$||y-\Gamma u||_2$")
        plt.ylabel("$||L u||_2$")
        plt.grid(which="both")
        plt.tight_layout()
        plt.show()

    if pickle_data:
        with open('estimates/step_experiment_l_curve.pickle', 'wb') as handle:
            pickle.dump([l_norm, residual_norm], handle, protocol=pickle.HIGHEST_PROTOCOL)


def ramp_excitation(sim_times, fs, bs, plot_input=False):
    U_ramp = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_ramp.shape[0])
    e2 = np.random.normal(0, .01, U_ramp.shape[0])

    U_ramp[1200:3200,0] += np.linspace(0, 2.7, 2000)
    U_ramp[3200:6200,0] += 2.7
    U_ramp[6200:8200,0] += np.flip(np.linspace(0, 2.7, 2000))
    U_ramp[:,0] += e1
    U_ramp[:,1] += e2

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_ramp[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_ramp[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_ramp


def step_excitation(sim_times, fs, bs, plot_input=False):
    U_step = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_step.shape[0])
    e2 = np.random.normal(0, .01, U_step.shape[0])

    U_step[:,0] += np.flip(np.linspace(-2.5, 2.5, 10000)) + e1
    U_step[:,1] += e2
    U_step[3200:5200,1] += 1.0
    U_step[5200:8200,1] -= 4.0
    U_step[8200:,1] += 0.5

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_step[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_step[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_step


def impulse_excitation(sim_times, fs, bs, plot_input=False):
    U_imp = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_imp.shape[0])
    e2 = np.random.normal(0, .01, U_imp.shape[0])

    one_hit = np.hstack((np.linspace(0, 0.34, 7), np.linspace(0.38, 0.64, 7), np.linspace(0.67, 0.87, 7), np.linspace(0.9, 0.98, 7)))

    U_imp[:,0] += 2.7 + e1
    U_imp[:,1] += e2
    U_imp[3200:3228,1] += one_hit
    U_imp[3228:3228+28,1] += np.flip(one_hit)
    U_imp[5200:5228,1] -= one_hit*1.2
    U_imp[5228:5228+28,1] -= np.flip(one_hit*1.2)
    U_imp[8200:8228,1] += one_hit*2
    U_imp[8228:8228+28,1] += np.flip(one_hit*2)

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_imp[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_imp[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_imp


def sinusoidal_excitation(sim_times, fs, bs, plot_input=False):

    def sum_sines(freqs, amps, phases, times, dc_offset):
        """
        Generates a sinusoidal signal that is a sum of three sine waves with different frequencies.

        Args:
        freqs (list[float]): List of three frequencies for the sine waves.
        amps (list[float]): List of three amplitudes for the sine waves.
        phases (list[float]): List of three phase shifts for the sine waves.
        times (numpy.ndarray): Timesteps of the signal in seconds.

        Returns:
        numpy.ndarray: Sinusoidal signal that is a sum of three sine waves.
        """

        signal = np.zeros(len(times))
        for f, a, p in zip(freqs, amps, phases):
            signal += a * np.sin(2 * np.pi * f * times + p)

        return signal + dc_offset

    freqs = [20, 40, 60]
    amps = [2, 1, 0.5]
    phases = [0, 0, 0]
    offset = 0
    sine_signal = sum_sines(freqs, amps, phases, sim_times, offset)

    U_sin = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .01, U_sin.shape[0])
    e2 = np.random.normal(0, .01, U_sin.shape[0])

    U_sin[:,0] += 2.7 + e1
    U_sin[:,1] += sine_signal + e2

    if plot_input:
        plt.subplot(211)
        plt.plot(sim_times, U_sin[:,0], label='Driving motor setpoint', color='blue')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.subplot(212)
        plt.plot(sim_times, U_sin[:,1], label='Loading motor setpoint', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return U_sin


def input_and_state_estimation(load, sim_times, batch_size, lam_tikh, lam_lasso, run_cvx=False, run_elastic_net=False, run_kf=False, use_trend_filter=False, use_virtual_sensor=True, pickle_results=False, fname='dummy.pickle'):
    dt = np.mean(np.diff(sim_times))
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    # include second torque transducer
    C_mod = np.insert(C, 3, np.zeros((1, C.shape[1])), 0)
    C_mod[3,22+18] += 2e4
    D_mod = np.zeros((C_mod.shape[0], B.shape[1]))

    tout, yout, _ = dlsim((A, B, C_mod, D_mod, dt), u=load, t=sim_times)
    e3 = np.random.normal(0, .01, yout.shape)
    y_noise = yout + e3

    n = len(tout)
    meas = y_noise
    motor = load[:,0]
    propeller = load[:,-1]

    if run_cvx:
        input_tikh, states_tikh = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            tout,
            lam=lam_tikh,
            use_trend_filter=use_trend_filter,
            use_virtual_sensor=use_virtual_sensor
        )

        input_lasso, states_lasso = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            tout,
            lam=lam_lasso,
            use_zero_init=True,
            use_lasso=True,
            use_trend_filter=use_trend_filter,
            use_virtual_sensor=use_virtual_sensor
        )

        ie.subplot_input_estimates(sim_times, motor, propeller, input_tikh, input_lasso, n, batch_size)

        if use_virtual_sensor:
            ie.plot_virtual_sensor(sim_times, yout[:,2:], states_tikh, states_lasso)

    if run_elastic_net:
        input_net, states_net = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            tout,
            lam=lam_lasso,
            use_zero_init=True,
            use_lasso=True,
            use_elastic_net=True,
            use_trend_filter=use_trend_filter,
            use_virtual_sensor=use_virtual_sensor
        )

    if run_kf:
        times_kf, input_estimates_kf, torque_estimates_kf = kf.run_kalman_filter(
            sim_times,
            np.vstack((meas[:,0], meas[:,1])),
            meas[:,2],
            meas[:,2:],
            load[:,0],
            load[:,-1],
            np.mean(load[:,0])
        )

    if pickle_results:
        with open(fname, 'wb') as handle:
            pickle.dump(
                [
                    sim_times,
                    load[:,0],
                    load[:,1],
                    input_tikh,
                    input_lasso,
                    meas,
                    states_tikh,
                    states_lasso,
                    input_estimates_kf,
                    torque_estimates_kf
                ],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)


def simulation_experiment():
    fs = 1000
    sim_times = np.arange(0, 10, 1/fs)
    dt = np.mean(np.diff(sim_times))
    batch_size = 500

    plot_input = False

    ramp_load = ramp_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    step_load = step_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    impulse_load = impulse_excitation(sim_times, fs, batch_size, plot_input=plot_input)
    sinusoidal_load = sinusoidal_excitation(sim_times, fs, batch_size, plot_input=plot_input)

    load = ramp_load

    plot_l_curve = 0
    run_estimation = 1

    if plot_l_curve:
        s, e = 0, 0
        plot_L_curve(
            sim_times[:batch_size],
            dt,
            load[s:e+batch_size],
            show_plot=True,
            pickle_data=False
        )

    if run_estimation:
        input_and_state_estimation(
            load,
            sim_times,
            batch_size,
            0.1,
            0.1,
            run_cvx=True,
            run_elastic_net=False,
            run_kf=True,
            use_trend_filter=True,
            use_virtual_sensor=True,
            pickle_results=True,
            fname='estimates/ramp_experiment_lam01_trend.pickle'
        )


if __name__ == "__main__":
    start, stop = 10000, 90000

    motor_filename = "../data/IceExcitation_1000-2000_1_motor.tdms"
    time_motor, motor_setpoint, motor, propeller_setpoint, propeller = read_motor_tdms(
        motor_filename,
        start,
        stop
    )

    sensor_filename = "../data/ramp_0.tdms"
    time, speed1, speed2, torque1, torque2 = read_sensor_tdms(sensor_filename, start, stop)

    plt.figure()
    plt.plot(time, speed1)

    plt.show()

    # TODO: do for measured data
    input_and_state_estimation(
        load,
        sim_times,
        500,
        0.1,
        0.1,
        run_cvx=True,
        run_elastic_net=False,
        run_kf=True,
        use_trend_filter=True,
        use_virtual_sensor=True,
        pickle_results=True,
        fname='estimates/ramp_experiment_lam01_trend.pickle'
    )

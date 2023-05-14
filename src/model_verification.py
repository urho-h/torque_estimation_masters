import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlsim, butter, lfilter

import testbench_MSSP as tb


def low_pass_filter(signal, cutoff, fs):
    '''
    A fifth-order Butterworth low-pass filter.
    '''
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(5, normalized_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def simulate(times, load, measurements):
    dt = np.mean(np.diff(times))
    A, B, C, D = get_testbench_state_space(dt)
    tout, yout, _ = dlsim((A, B, C, D, dt), u=load, t=times)

    plt.plot(tout, yout[:,-1], label="sim")
    plt.plot(measurements[:,0], measurements[:,-2], label="meas")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_sensor = False
    plot_motor = False
    run_simulation = False
    save_processed_data = True

    sensor_data = np.loadtxt(
        "../data/masters_data/raw_data/impulse_sensor.csv",
        delimiter=",",
        skiprows=1
    )

    time_raw = sensor_data[:,0]*25e-9
    time = time_raw-time_raw[0]

    enc1_angle = (sensor_data[:,1])*(2*np.pi/20000)
    enc1_time_raw = sensor_data[:,2]*25e-9
    enc1_time = enc1_time_raw - enc1_time_raw[0]
    enc2_angle = (sensor_data[:,3])*(2*np.pi/20000)
    enc2_time_raw = sensor_data[:,4]*25e-9
    enc2_time = enc2_time_raw - enc2_time_raw[0]

    speed1 = np.gradient(enc1_angle, enc1_time)
    speed2 = np.gradient(enc2_angle, enc2_time)

    # TODO: torque sensor data acquisition got switched up
    torque2 = sensor_data[:,-2]*10 # actually torque1 in the dataset
    torque1 = sensor_data[:,-1]*4 # actually torque2 in the dataset

    # lowpass filter
    speed1 = low_pass_filter(speed1, 500, 3012)
    speed2 = low_pass_filter(speed2, 500, 3012)
    torque1 = low_pass_filter(torque1, 500, 3012)
    torque2 = low_pass_filter(torque2, 500, 3012)

    if plot_sensor:
        plt.figure()
        plt.plot(time, speed1)
        plt.figure()
        plt.plot(time, speed2)
        plt.figure("torque1")
        plt.plot(time, torque1)
        plt.figure("torque2")
        plt.plot(time, torque2)

        plt.show()

    measurements = np.vstack((time[::3], speed1[::3], speed2[::3], torque1[::3], torque2[::3])).T

    motor_data = np.loadtxt("../data/masters_data/raw_data/impulse_motor.csv", delimiter=",", skiprows=1)

    time_motor = motor_data[:,0]
    motor_set = motor_data[:,1]
    motor = motor_data[:,2]
    propeller_set = motor_data[:,5]*8
    propeller = motor_data[:,6]*8

    if plot_motor:
        plt.figure()
        plt.plot(time_motor, motor_set)
        plt.figure()
        plt.plot(time_motor, motor)
        plt.figure()
        plt.plot(time_motor, propeller_set)
        plt.figure()
        plt.plot(time_motor, propeller)

        plt.show()

    motor_measurements = np.vstack(
        (time_motor,
         motor_set,
         motor,
         propeller_set,
         propeller)
    )

    if run_simulation:
        load = np.vstack((motor, propeller)).T
        simulate(time_motor, load, measurements)

    if save_processed_data:
        np.savetxt(
            "../data/masters_data/processed_data/impulse_sensor.csv",
            measurements,
            delimiter=","
        )
        np.savetxt(
            "../data/masters_data/processed_data/impulse_motor.csv",
            motor_measurements,
            delimiter=","
        )

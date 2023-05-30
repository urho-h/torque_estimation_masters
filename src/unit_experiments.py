import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlsim
import pickle

import testbench_MSSP as tb
from convex_optimization import input_estimation as ie
from kalmanfilter import kalmanfilter as kf


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 12,
    "figure.figsize": (6,4),
})


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def input_and_state_estimation(load, meas, sim_times, batch_size, lam_tikh, lam_lasso, overlap=50, run_tikh=False, run_lasso=False, run_elastic_net=False, run_kf=False, use_trend_filter=False, pickle_results=False, fname='estimation_results.pickle'):
    dt = np.mean(np.diff(sim_times))
    A, B, C, D = get_testbench_state_space(dt)
    sys = (A, B, C, D)

    # include second torque transducer
    C_mod = np.insert(C, 3, np.zeros((1, C.shape[1])), 0)
    C_mod[3,22+18] += 2e4
    D_mod = np.zeros((C_mod.shape[0], B.shape[1]))

    n = len(sim_times)
    motor = load[:,0]
    propeller = load[:,-1]

    if run_tikh:
        input_tikh = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            overlap,
            sim_times,
            lam=lam_tikh,
            use_trend_filter=use_trend_filter,
            pickle_data=pickle_results,
            fn=fname
        )

    if run_lasso:
        input_lasso = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            overlap,
            sim_times,
            lam=lam_lasso,
            use_zero_init=True,
            use_lasso=True,
            use_trend_filter=use_trend_filter,
            pickle_data=pickle_results,
            fn=fname
        )

    if run_elastic_net:
        input_net = ie.estimate_input(
            sys,
            meas[:,:3],
            batch_size,
            overlap,
            sim_times,
            lam=lam_tikh,
            lam2=lam_lasso,
            use_zero_init=True,
            use_lasso=False,
            use_elastic_net=True,
            use_trend_filter=use_trend_filter,
            pickle_data=pickle_results,
            fn=fname
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


def measurements_experiment(run_input_estimation=False):
    sensor_data = np.loadtxt("../data/masters_data/processed_data/CFD_2000_sensor.csv", delimiter=",")
    motor_data = np.loadtxt("../data/masters_data/processed_data/CFD_2000_motor.csv", delimiter=",")
    time = sensor_data[:,0]

    measurements = sensor_data[:,1:]

    load = np.vstack((motor_data[:,2], motor_data[:,4])).T

    if run_input_estimation:
        input_and_state_estimation(
            load,
            measurements,
            time[:measurements.shape[0]],
            500,
            0.01,
            10,
            run_tikh=False,
            run_lasso=True,
            use_trend_filter=True,
            pickle_results=True,
            fname='estimates/CFD_2000_experiment/l1_trend_lam10_'
        )


if __name__ == "__main__":
    measurements_experiment(run_input_estimation=True)

    # DONE: impulse excitation
    # DONE: sinusoidal excitation
    # DONE: step excitation
    # DONE: ramp excitation
    # DONE: ice excitation
    # DONE: CFD excitation

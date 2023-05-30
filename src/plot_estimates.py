import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
from scipy.signal import dlsim

import testbench_MSSP as tb


plt.style.use('science')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 12,
    "figure.figsize": (8,6),
})


def get_testbench_state_space(dt):
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def get_states(fn, n_batches, case, sensor_data):
    motor_estimates, propeller_estimates = [], []
    overlap = 50

    for i in range(n_batches):
        with open(fn + str(i) + ".pickle", 'rb') as handle:
            dataset = pickle.load(handle)
            if i == 0:
                all_motor_estimates = dataset[0][::2]
                motor_estimates = all_motor_estimates[:-2*overlap]
                all_propeller_estimates = dataset[0][1::2]
                propeller_estimates = all_propeller_estimates[:-2*overlap]
            else:
                all_motor_estimates = dataset[0][::2]
                motor_estimates = np.concatenate(
                    (motor_estimates, all_motor_estimates[overlap:-overlap])
                )
                all_propeller_estimates = dataset[0][1::2]
                propeller_estimates = np.concatenate(
                    (propeller_estimates, all_propeller_estimates[overlap:-overlap])
                )

    time = sensor_data[:,0]
    dt = np.mean(np.diff(time))
    t_estimate = np.linspace(0, len(propeller_estimates)*dt, len(propeller_estimates))

    U_est = np.hstack((motor_estimates, propeller_estimates))

    A, B, C, D = get_testbench_state_space(dt)
    C_mod = np.insert(C, C.shape[0], np.zeros((1, C.shape[1])), 0)
    C_mod[C.shape[0],22+18] += 2e4

    tout_ests, yout_ests, _ = dlsim(
        (A, B, C_mod, np.zeros((C_mod.shape[0], B.shape[1])), dt),
        U_est,
        t=t_estimate
    )

    return t_estimate, yout_ests[:-1,-1]


def plot_maritime_estimates():
    case = "ice_2000"
    # n = 32
    n = 20
    sensor_data_ice = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    # tet, det = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_ice)
    # tel1, del1 = get_states("estimates/" + case + "_experiment/l1_lam1_", n, case, sensor_data_ice)
    tehp, dehp = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_ice)
    tel1t, del1t = get_states("estimates/" + case + "_experiment/l1_trend_lam1_", n, case, sensor_data_ice)

    plt.figure()
    plt.plot(sensor_data_ice[:,0], sensor_data_ice[:,-1], color='black', label='Measurement')
    # plt.plot(tet, det, color='dimgray', label='Tikhonov regularization')
    # plt.plot(tel1, del1, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
    plt.plot(tehp, dehp, color='red', label='H-P trend filtering')
    # plt.plot(tel1t, del1t, color='blue', label='$\ell_1$ trend filtering')
    # plt.xlim(7.4,8.7)
    plt.xlim(3.8,5)
    plt.ylim(2,24)
    plt.legend()
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")
    # plt.savefig("../figures/ice_2000_torque_estimate.pdf")
    plt.show()

    case = "CFD_2000"
    # n = 71
    n = 76
    sensor_data_CFD = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    # tet, det = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_CFD)
    # tel1, del1 = get_states("estimates/" + case + "_experiment/l1_lam1_", n, case, sensor_data_CFD)
    # tehp, dehp = get_states("estimates/" + case + "_experiment/hp_trend_lam10_", n, case, sensor_data_CFD)
    tel1t, del1t = get_states("estimates/" + case + "_experiment/l1_trend_lam10_", n, case, sensor_data_CFD)

    plt.figure()
    plt.plot(sensor_data_CFD[:,0], sensor_data_CFD[:,-1], color='black', label='Measurement')
    # plt.plot(tet, det, color='dimgray', label='Tikhonov regularization')
    # plt.plot(tel1, del1, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
    # plt.plot(tehp, dehp, color='red', label='H-P trend filtering')
    plt.plot(tel1t, del1t, color='blue', label='$\ell_1$ trend filtering')
    plt.xlim(4,34)
    plt.ylim(0,21)
    plt.legend()
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")
    # plt.savefig("../figures/CFD_2000_torque_estimate.pdf")
    plt.show()


def plot_unit_estimates():
    case = "impulse"
    n = 10
    sensor_data_impulse = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    tet, det = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_impulse)
    tel1, del1 = get_states("estimates/" + case + "_experiment/l1_lam1_", n, case, sensor_data_impulse)
    tehp, dehp = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_impulse)
    tel1t, del1t = get_states("estimates/" + case + "_experiment/l1_trend_lam10_", n, case, sensor_data_impulse)

    ax1 = plt.subplot(221)
    # plt.title("a)", loc='left')
    ax1.plot(sensor_data_impulse[:,0], sensor_data_impulse[:,-1], color='black', label='Measurement')
    ax1.plot(tet, det, color='dimgray', label='Tikhonov regularization')
    ax1.plot(tel1, del1, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
    ax1.plot(tehp, dehp, color='red', label='H-P trend filtering')
    ax1.plot(tel1t, del1t, color='blue', label='$\ell_1$ trend filtering')
    ax1.set_xlim(1.1,1.6)
    ax1.set_ylim(2,21)
    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(1.1, 1.3),
        fancybox=True,
        shadow=True,
        ncol=3
    )
    ax1.set_ylabel("Torque (Nm)")
    ax1.set_xlabel("Time (s)")

    case = "sin"
    n = 18
    sensor_data_sin = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    tets, dets = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_sin)
    tel1s, del1s = get_states("estimates/" + case + "_experiment/l1_lam01_", n, case, sensor_data_sin)
    tehps, dehps = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_sin)
    tel1ts, del1ts = get_states("estimates/" + case + "_experiment/l1_trend_lam1_", n, case, sensor_data_sin)

    plt.subplot(222)
    # plt.title("b)", loc='left')
    plt.plot(sensor_data_sin[:,0], sensor_data_sin[:,-1], color='black', label='Torque transducer 2')
    plt.plot(tets, dets, color='dimgray', label='Tikhonov regularization')
    plt.plot(tel1s, del1s, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
    plt.plot(tehps, dehps, color='red', label='H-P trend filtering')
    plt.plot(tel1ts, del1ts, color='blue', label='$\ell_1$ trend filtering')
    plt.xlim(1.49,1.55)
    plt.ylim(-1,31)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    case = "step"
    n = 25
    sensor_data_step = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    tetst, detst = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_step)
    tel1st, del1st = get_states("estimates/" + case + "_experiment/l1_lam1_", n, case, sensor_data_step)
    tehpst, dehpst = get_states("estimates/" + case + "_experiment/hp_trend_lam01_", n, case, sensor_data_step)
    tel1tst, del1tst = get_states("estimates/" + case + "_experiment/l1_trend_lam10_", n, case, sensor_data_step)

    plt.subplot(223)
    # plt.title("c)", loc='left')
    plt.plot(sensor_data_step[:,0], sensor_data_step[:,-1], color='black', label='Torque transducer 2')
    plt.plot(tetst, detst, color='dimgray', label='Tikhonov regularization')
    plt.plot(tel1st, del1st, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
    plt.plot(tehpst, dehpst, color='red', label='H-P trend filtering')
    plt.plot(tel1tst, del1tst, color='blue', label='$\ell_1$ trend filtering')
    plt.xlim(4.2,5.3)
    plt.ylim(-1,27)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    case = "ramp"
    n = 136
    sensor_data_ramp = np.loadtxt("../data/masters_data/processed_data/" + case + "_sensor.csv", delimiter=",")

    tetr, detr = get_states("estimates/" + case + "_experiment/tikh_lam001_", n, case, sensor_data_ramp)
    tel1r, del1r = get_states("estimates/" + case + "_experiment/l1_lam1_", n, case, sensor_data_ramp)
    tehpr, dehpr = get_states("estimates/" + case + "_experiment/hp_trend_lam10_", n, case, sensor_data_ramp)
    tel1tr, del1tr = get_states("estimates/" + case + "_experiment/l1_trend_lam10_", n, case, sensor_data_ramp)

    plt.subplot(224)
    # plt.title("d)", loc='left')
    plt.plot(sensor_data_ramp[:,0], sensor_data_ramp[:,-1], color='black', label='Torque transducer 2')
    plt.plot(tetr, detr, color='dimgray', label='Tikhonov regularization')
    plt.plot(tel1r, del1r, color='dimgray', linestyle='dashed', label='$\ell_1$-regularization')
    plt.plot(tehpr, dehpr, color='red', label='H-P trend filtering')
    plt.plot(tel1tr, del1tr, color='blue', label='$\ell_1$ trend filtering')
    plt.xlim(34.2,35.8)
    plt.ylim(2,9)
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time (s)")

    # plt.savefig("../figures/unit_torque_estimates.pdf")
    plt.show()

if __name__ == "__main__":
    # plot_unit_estimates()
    plot_maritime_estimates()

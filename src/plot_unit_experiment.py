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
    "figure.figsize": (6,4),
})


def get_testbench_state_space(dt):
    # inertias, stiffs, damps, damps_ext, ratios = tb.parameters()
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios)

    A, B = tb.c2d(Ac, Bc, dt)

    return A, B, C, D


def plot_batch(fn, n_batches):
    motor_data = np.loadtxt("../data/masters_data/processed_data/sin_148_motor.csv", delimiter=",")
    time = motor_data[:,0]
    motor = motor_data[:,2]
    propeller = motor_data[:,-1]

    # time = np.loadtxt("../data/ice_excitation/times.csv", delimiter=",")
    # motor = np.loadtxt("../data/ice_excitation/motor.csv", delimiter=",")
    # propeller = np.loadtxt("../data/ice_excitation/propeller.csv", delimiter=",")

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
                print(propeller_estimates.shape)
            else:
                all_motor_estimates = dataset[0][::2]
                motor_estimates = np.concatenate(
                    (motor_estimates, all_motor_estimates[overlap:-overlap])
                )
                all_propeller_estimates = dataset[0][1::2]
                propeller_estimates = np.concatenate(
                    (propeller_estimates, all_propeller_estimates[overlap:-overlap])
                )

    # t_estimate = np.linspace(0, time[:propeller_estimates.shape[0]], propeller_estimates.shape[0])
    t_estimate = time

    plt.figure()
    plt.plot(t_estimate, propeller_estimates, color='red', label='estimate')
    plt.plot(time, propeller, color='blue', label='Propeller torque')
    plt.legend()

    plt.figure()
    plt.plot(t_estimate, motor_estimates, color='red', label='estimate')
    plt.plot(time, motor, color='blue', label='Motor torque')
    plt.legend()

    #######################################
    sensor_data = np.loadtxt("../data/masters_data/processed_data/sin_148_sensor.csv", delimiter=",")
    # sensor_speed = np.loadtxt("../data/ice_excitation/speed_measurements.csv", delimiter=",")
    # sensor_torque = np.loadtxt("../data/ice_excitation/torque_measurements.csv", delimiter=",")
    # torque2 = np.genfromtxt("../data/ice_excitation/torques.csv", delimiter=",", usecols=(18)).T
    # sensor_data = np.vstack((time, sensor_speed, sensor_torque, torque2)).T

    time = sensor_data[:,0]
    dt = np.mean(np.diff(time))

    U_est = np.hstack((motor_estimates, propeller_estimates))

    A, B, C, D = get_testbench_state_space(dt)
    C_mod = np.insert(C, C.shape[0], np.zeros((1, C.shape[1])), 0)
    C_mod[C.shape[0],22+18] += 2e4

    tout_ests, yout_ests, _ = dlsim(
        (A, B, C_mod, np.zeros((C_mod.shape[0], B.shape[1])), dt),
        U_est,
        t=t_estimate
    )

    plt.figure()
    plt.plot(time, sensor_data[:,-1], color='blue', label='Torque transducer 2')
    plt.plot(np.linspace(0, time[-1], yout_ests.shape[0]), yout_ests[:,-1], color='red', label='estimate')
    plt.legend()
    plt.show()


def plot_l_curve_impulse():
    with open("estimates/pareto_curves/impulse_experiment_tikh_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_tikh = dataset[0]
        residual_norm_tikh = dataset[1]
        lambdas_tikh = dataset[2]

    with open("estimates/pareto_curves/impulse_experiment_l1_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1 = dataset[0]
        residual_norm_l1 = dataset[1]
        lambdas_l1 = dataset[2]

    with open("estimates/pareto_curves/impulse_experiment_hp_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_hp_trend = dataset[0]
        residual_norm_hp_trend = dataset[1]
        lambdas_hp_trend = dataset[2]

    with open("estimates/pareto_curves/impulse_experiment_l1_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1_trend = dataset[0]
        residual_norm_l1_trend = dataset[1]
        lambdas_l1_trend = dataset[2]

    plt.subplot(221)
    plt.title("a)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_tikh[:-3], l_norm_tikh[:-3], color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_tikh[4]),
        (residual_norm_tikh[4], l_norm_tikh[4]),
        (residual_norm_tikh[4]+100, l_norm_tikh[4])
    )
    plt.ylabel("$||L u||_2$")

    plt.subplot(222)
    plt.title("b)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_hp_trend, l_norm_hp_trend, color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[5]),
        (residual_norm_hp_trend[5], l_norm_hp_trend[5]),
        (residual_norm_hp_trend[5]+1, l_norm_hp_trend[5]+0.5)
    )

    plt.subplot(223)
    plt.title("c)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1, l_norm_l1, color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1[6]),
        (residual_norm_l1[6], l_norm_l1[6]),
        (residual_norm_l1[6]+0.5, l_norm_l1[6]+1)
    )
    plt.xlabel("$||y-\Gamma u||_2$")
    plt.ylabel("$||L u||_1$")

    plt.subplot(224)
    plt.title("d)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1_trend, l_norm_l1_trend, color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1_trend[9]),
        (residual_norm_l1_trend[9], l_norm_l1_trend[9]),
        (residual_norm_l1_trend[9]+0.5, l_norm_l1_trend[9]+0.5)
    )
    plt.xlabel("$||y-\Gamma u||_2$")

    plt.tight_layout()
    # plt.savefig("../figures/l_curve_impulse.pdf")
    plt.show()


def plot_l_curve_sinusoidal():
    with open("estimates/pareto_curves/sin_experiment_tikh_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_tikh = dataset[0]
        residual_norm_tikh = dataset[1]
        lambdas_tikh = dataset[2]

    with open("estimates/pareto_curves/sin_experiment_l1_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1 = dataset[0]
        residual_norm_l1 = dataset[1]
        lambdas_l1 = dataset[2]

    with open("estimates/pareto_curves/sin_experiment_hp_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_hp_trend = dataset[0]
        residual_norm_hp_trend = dataset[1]
        lambdas_hp_trend = dataset[2]

    with open("estimates/pareto_curves/sin_experiment_l1_trend_curve.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        l_norm_l1_trend = dataset[0]
        residual_norm_l1_trend = dataset[1]
        lambdas_l1_trend = dataset[2]

    plt.subplot(221)
    plt.title("a)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_tikh[:-3], l_norm_tikh[:-3], color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_tikh[4]),
        (residual_norm_tikh[4], l_norm_tikh[4]),
        (residual_norm_tikh[4]+100, l_norm_tikh[4]+1)
    )
    plt.ylabel("$||L u||_2$")

    plt.subplot(222)
    plt.title("b)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_hp_trend, l_norm_hp_trend, color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_hp_trend[6]),
        (residual_norm_hp_trend[6], l_norm_hp_trend[6]),
        (residual_norm_hp_trend[6]+10, l_norm_hp_trend[6]+1)
    )

    plt.subplot(223)
    plt.title("c)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1, l_norm_l1, color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1[6]),
        (residual_norm_l1[6], l_norm_l1[6]),
        (residual_norm_l1[6]+1, l_norm_l1[6]+1)
    )
    plt.xlabel("$||y-\Gamma u||_2$")
    plt.ylabel("$||L u||_1$")

    plt.subplot(224)
    plt.title("d)")
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(residual_norm_l1_trend, l_norm_l1_trend, color='blue')
    plt.annotate(
        "$\lambda$ = " + str(lambdas_l1_trend[9]),
        (residual_norm_l1_trend[9], l_norm_l1_trend[9]),
        (residual_norm_l1_trend[9]+1, l_norm_l1_trend[9]+1)
    )
    plt.xlabel("$||y-\Gamma u||_2$")

    plt.tight_layout()
    # plt.savefig("../figures/l_curve_sinusoidal.pdf")
    plt.show()


if __name__ == "__main__":
    # plot_batch("estimates/sin_test_short/sin_experiment_tikh_lam0005_", 32)
    # plot_l_curve_sinusoidal()
    plot_l_curve_impulse()

    # TODO: calculate l curves using motor setpoint data

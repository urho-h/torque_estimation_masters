import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

import testbench_MSSP as tb


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 11,
})


def get_state_matrices():
    inertias, stiffs, damps, damps_ext, ratios = tb.new_parameters()
    Ac, Bc, C, D = tb.state_space_matrices(inertias, stiffs, damps, damps_ext, ratios, full_B=True)

    return Ac, Bc


def modal_analysis(A, B):
    lam, vec = LA.eig(A)

    lam = lam[::2]
    omegas = np.sort(np.absolute(lam))
    omegas_damped = np.sort(np.abs(np.imag(lam)))
    freqs = omegas / (2 * np.pi)
    print(freqs)

    vec = vec[: int(vec.shape[0] / 2)]
    vec = vec[:, ::2]
    inds = np.argsort(np.abs(lam))
    eigenmodes = np.zeros(vec.shape)
    for i, v in enumerate(inds):
        eigenmodes[:, i] = vec[:, v]

    # rigid body mode
    # plt.figure()
    # plt.scatter(np.arange(1, 22, 1), eigenmodes[:,0], label="$f_1$ = " + str(freqs[0].round(1)) + "Hz", color="black")

    plt.figure()
    plt.subplot(311)
    plt.scatter(np.arange(1, 22, 1), eigenmodes[:,1], label="$f_1$ = " + str(freqs[1].round(1)) + "Hz", color="black")
    plt.plot(np.arange(1, 13, 1), eigenmodes[:12,1], color="black")
    plt.plot(np.arange(13, 18, 1), eigenmodes[12:17,1], color="black")
    plt.plot(np.arange(18, 22, 1), eigenmodes[17:,1], color="black")
    plt.legend()
    plt.xticks(range(1, 22))
    plt.tick_params('y', labelleft=False, left=False)


    plt.subplot(312)
    plt.scatter(np.arange(1, 22, 1), eigenmodes[:,2], label="$f_2$ = " + str(freqs[2].round(1)) + "Hz", color="black")
    plt.plot(np.arange(1, 13, 1), eigenmodes[:12,2], color="black")
    plt.plot(np.arange(13, 18, 1), eigenmodes[12:17,2], color="black")
    plt.plot(np.arange(18, 22, 1), eigenmodes[17:,2], color="black")
    plt.legend()
    plt.xticks(range(1, 22))
    plt.tick_params('y', labelleft=False, left=False)

    plt.subplot(313)
    plt.scatter(np.arange(1, 22, 1), eigenmodes[:,3], label="$f_3$ = " + str(freqs[3].round(1)) + "Hz", color="black")
    plt.plot(np.arange(1, 13, 1), eigenmodes[:12,3], color="black")
    plt.plot(np.arange(13, 18, 1), eigenmodes[12:17,3], color="black")
    plt.plot(np.arange(18, 22, 1), eigenmodes[17:,3], color="black")
    plt.legend()
    plt.xticks(range(1, 22))
    plt.tick_params('y', labelleft=False, left=False)
    plt.xlabel("Node number")

    plt.tight_layout()
    plt.savefig("../figures/mode_shapes.pdf")
    plt.show()


def run_fft(time, torques):
    Fs = 1/np.mean(np.diff(time))  # Sampling frequency
    torques -= np.mean(torques)
    L = len(torques)
    Y = np.fft.fft(torques)
    P2 = abs(Y/L)
    P1 = P2[0:int(L/2)]
    P1[1:-2] = 2*P1[1:-2]
    f = (Fs/L)*np.linspace(0, (L/2), int(L/2))

    return f, P1


def plot_PRBS_result():
    sensor_data = np.loadtxt("../data/masters_data/processed_data/2000rpm_PRBS_sensor.csv", delimiter=",")

    time = sensor_data[:-3000,0]
    torque = sensor_data[3000:,-1]

    f, P1 = run_fft(time, torque)

    plt.plot(f, P1, color='blue')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Nm)")
    plt.xlim(0, 80)
    plt.grid()

    plt.tight_layout()
    # plt.savefig("../figures/2000rpm_PRBS_result.pdf")
    plt.show()


if __name__ == "__main__":
    A, B = get_state_matrices()
    modal_analysis(A, B)
    plot_PRBS_result()

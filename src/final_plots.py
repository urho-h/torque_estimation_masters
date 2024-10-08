import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.linalg as LA
from scipy.signal import dlsim, butter, lfilter, welch
import scienceplots

import opentorsion as ot

import testbench_model_minimal as tbm
import data_equation as deq


np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100'})


def impulse_excitation(sim_times, plot=False):
    """
    Parameters:

    sim_times : numpy.ndarray
        Timesteps of the simulation.

    Returns:

    U_imp : numpy.ndarray
        Input torque matrix. Rows correspond to timesteps, first column has the motor torque,
        second column has the propeller torque.
    """
    U_imp = np.zeros((len(sim_times), 2))

    e1 = np.random.normal(0, .1, U_imp.shape[0])
    e2 = np.random.normal(0, .1, U_imp.shape[0])

    one_hit = -np.hstack((np.linspace(0, 0.34, 7), np.linspace(0.38, 0.64, 7), np.linspace(0.67, 0.87, 7), np.linspace(0.9, 0.98, 7)))

    U_imp[:,0] += 2.7 + e1
    U_imp[:,1] += e2
    U_imp[3200:3228,1] += one_hit*10
    U_imp[3228:3228+28,1] += np.flip(one_hit*10)
    U_imp[5200:5228,1] += one_hit*10
    U_imp[5228:5228+28,1] += np.flip(one_hit*10)
    U_imp[8200:8228,1] += one_hit*10
    U_imp[8228:8228+28,1] += np.flip(one_hit*10)

    if plot:
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


def simulated_impulse_experiment(show_plot=False):
    fs = 1000  # sampling frequency
    t_span = (0,10)  # simulation start and end times
    t_sim = np.arange(t_span[0], t_span[1], 1/fs)  # timesteps
    dt_sim = np.mean(np.diff(t_sim))  # timestep length
    impulse_sim = impulse_excitation(t_sim)

    # excitation with one impulse
    t_sim_short = t_sim[:1000]
    excitation_sim = impulse_sim[2500:3500]

    A, B, C, D, c, d = tbm.get_testbench_state_space(dt_sim)
    ss = (A, B, c, d, dt_sim)
    ss2 = (A, B, C, D, dt_sim)

    tout_sim, yout_sim, xout_sim = dlsim(ss2, excitation_sim, t=t_sim_short)  # simulate

    measurements = yout_sim + np.random.normal(0, .1, yout_sim.shape)  # add measurement noise

    # estimation parameters
    lam1 = 1e-1
    lam2 = 10
    batch_size = 500

    # call estimation function
    phi_sim, state_ests = deq.ell2_analytical(
        ss,
        ss2,
        measurements[:,:-1],
        batch_size,
        0,
        t_sim_short,
        lam1=lam1,
        lam2=lam2,
        use_trend_filter=True
    )

    if show_plot:
        # Create a figure
        fig = plt.figure(figsize=(6, 4))

        # Define the grid layout with 3 rows and 2 columns
        gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 0.5, 1])

        # Create the first four subplots (2x2 grid)
        ax1 = fig.add_subplot(gs[0, 0])  # Row 0, Column 0
        ax2 = fig.add_subplot(gs[0, 1])  # Row 0, Column 1
        ax3 = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
        ax4 = fig.add_subplot(gs[1, 1])  # Row 1, Column 1

        # Create the bottom subplot that spans both columns
        ax5 = fig.add_subplot(gs[2, :])  # Row 2, spans all columns

        # Plot data
        ax1.plot(np.arange(0, 21, 1), xout_sim[500,:][:21], label=r'True $\tau(0)$ sim.')
        ax1.plot(np.arange(0, 21, 1), phi_sim[1][:21], 'b-', label=r'Estimated $\tau(0)$ sim.')
        ax1.set_xlabel("Node (-)", fontsize=11)
        ax1.set_ylabel("Torque (Nm)", fontsize=11)
        ax1.legend()
        ax1.minorticks_off()
        ax2.plot(np.arange(22, 44, 1), xout_sim[500,:][21:], label=r'True $\dot{\theta}(0)$ sim.')
        ax2.plot(np.arange(22, 44, 1), phi_sim[1][21:43], 'b-', label=r'Estimated $\dot{\theta}(0)$ sim.')
        ax2.set_xlabel("Node (-)", fontsize=11)
        ax2.set_ylabel("Speed (rad/s)", fontsize=11)
        ax2.legend()
        ax2.minorticks_off()

        motor_estimates_sim = np.concatenate((phi_sim[0][43::2], phi_sim[1][43::2]))
        propeller_estimates_sim = np.concatenate((phi_sim[0][44::2], phi_sim[1][44::2]))
        ax3.plot(t_sim_short, excitation_sim[:,0], label=r'True $u_{\mathrm{m}}$ sim.')
        ax3.plot(t_sim_short, motor_estimates_sim, 'b-', label=r'Estimated $u_{\mathrm{m}}$ sim.')#'Simulated driving motor input estimate')
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Torque (Nm)", fontsize=11)
        ax3.set_ylim(2.2, 4.3)
        ax3.legend()
        ax3.minorticks_off()
        ax4.plot(t_sim_short, excitation_sim[:,1], label=r'True $u_{\mathrm{p}}$ sim.')
        ax4.plot(t_sim_short, propeller_estimates_sim, 'b-', label=r'Estimated $u_{\mathrm{p}}$ sim.')
        ax4.set_xlabel("Time (s)", fontsize=11)
        ax4.set_ylabel("Torque (Nm)", fontsize=11)
        ax4.legend()
        ax4.minorticks_off()

        prop_shaft_estimates = np.concatenate((state_ests[0][3::4], state_ests[1][3::4]))
        ax5.plot(t_sim_short, yout_sim[:,-1], label="True simulated propeller shaft torque")
        ax5.plot(t_sim_short, prop_shaft_estimates, '-', c='b', label="Simulated propeller shaft torque estimate")
        ax5.set_xlabel("Time (s)", fontsize=11)
        ax5.set_ylabel("Torque (Nm)", fontsize=11)
        ax5.legend()
        ax5.minorticks_off()

        # Add some titles or labels for clarity
        ax1.set_title('a)', loc='left', fontsize=12)
        ax2.set_title('b)', loc='left', fontsize=12)
        ax3.set_title('c)', loc='left', fontsize=12)
        ax4.set_title('d)', loc='left', fontsize=12)
        ax5.set_title('e)', loc='left', fontsize=12)

        # Adjust the layout to prevent overlap
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # Show the plot
        # plt.savefig("uh1_figures/simulated_impulse_estimates.pdf")
        plt.show()

    return


def process_estimates(n_batches, estimates, nstates=43):
    """
    Here the input and initial state estimates are processed.
    Overlapped sections are discarded and the input estimate batches are stacked one after the other.
    """
    motor_estimates, propeller_estimates = [], []
    motor_est_overlap, prop_est_overlap = [], []
    for i in range(n_batches):
        if i == 0:
            all_motor_estimates = estimates[i][nstates::2]
            motor_est_overlap.append(all_motor_estimates)
            motor_estimates = all_motor_estimates[:-2]
            all_propeller_estimates = estimates[i][(nstates+1)::2]
            prop_est_overlap.append(all_propeller_estimates)
            propeller_estimates = all_propeller_estimates[:-2]
        else:
            all_motor_estimates = estimates[i][nstates::2]
            motor_est_overlap.append(all_motor_estimates)
            motor_estimates = np.concatenate(
                (motor_estimates, all_motor_estimates[1:-1])
            )
            all_propeller_estimates = estimates[i][(nstates+1)::2]
            prop_est_overlap.append(all_propeller_estimates)
            propeller_estimates = np.concatenate(
                (propeller_estimates, all_propeller_estimates[1:-1])
            )

    return motor_estimates, propeller_estimates, estimates[i][:44]


def ice_excitation_experiment(show_plot=False):
    dt = 1e-3
    A, B, C, D, c, d = tbm.get_testbench_state_space(dt)
    ss = (A, B, c, d, dt)
    ss2 = (A, B, C, D, dt)

    # The motor dataset columns:
    # | time | motor setpoint | motor | propeller setpoint | propeller |
    motor_data = np.loadtxt("../data/masters_data/processed_data/ice_2000_motor.csv", delimiter=",")

    # The sensor dataset columns:
    # | time | speed1 (encoder1) | speed2 (encoder2) | torque1 | torque2 |
    sensor_data = np.loadtxt("../data/masters_data/processed_data/ice_2000_sensor.csv", delimiter=",")
    time = sensor_data[:,0]
    measurements = sensor_data[:,1:] # measurement data from the encoder 1, encoder 2, torque transducer 1 and torque transducer 2
    motors = np.vstack((motor_data[:,2], motor_data[:,4])).T # motor torque data

    motor_setpoint = motor_data[:,1]
    propeller_setpoint = motor_data[:,3]
    motor = motors[:,0]
    propeller = motors[:,-1]

    lam_imp1 = 1e-2
    lam_imp2 = 10
    batch_size = 500
    estimates_imp, state_imp = deq.ell2_analytical(
        ss,
        ss2,
        measurements[:,:3],
        batch_size,
        1,
        time,
        lam1=lam_imp1,
        lam2=lam_imp2,
        use_trend_filter=True
    )

    ice_state_estimates = np.concatenate(state_imp)
    t2_estimates = ice_state_estimates[3::4]
    motor_estimates_imp, propeller_estimates_imp, final_x = process_estimates(
        len(estimates_imp),
        estimates_imp
    )

    if show_plot:
        fig = plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(2, 2, height_ratios=[0.5, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Row 1, Column 1
        ax2 = fig.add_subplot(gs[0, 1])  # Row 1, Column 2
        ax5 = fig.add_subplot(gs[1, :])  # Row 2, spans all columns

        ax1.plot(motor_data[:,0], motor_data[:,2], color='black', label=r'Measured $u_{\mathrm{m}}$')
        ax1.plot(time[:-(67+665)], motor_estimates_imp[665:], 'b-', label=r'Estimated $u_{\mathrm{m}}$')
        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_ylabel("Torque (Nm)", fontsize=11)
        ax1.set_xlim(3.7,5.2)
        ax1.legend()
        ax1.minorticks_off()

        ax2.plot(motor_data[:,0], motor_data[:,4], c='black', label=r'Measured $u_{\mathrm{p}}$')
        ax2.plot(time[:-(67+665)], -propeller_estimates_imp[665:], 'b-', label=r'Estimated $u_{\mathrm{p}}$')
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.set_ylabel("Torque (Nm)", fontsize=11)
        ax2.set_xlim(3.7,5.2)
        ax2.legend()
        ax2.minorticks_off()

        ax5.plot(time, measurements[:,-1], c='black', label=r'Measured $\tau_{20}$')
        ax5.plot(time[:-42], t2_estimates[15:], 'b-', label=r'Estimated $\tau_{20}$')
        ax5.set_xlabel("Time (s)", fontsize=11)
        ax5.set_ylabel("Torque (Nm)", fontsize=11)
        ax5.set_xlim(3.7,5.2)
        ax5.legend()
        ax5.minorticks_off()

        # Add some titles or labels for clarity
        ax1.set_title('a)', loc='left', fontsize=12)
        ax2.set_title('b)', loc='left', fontsize=12)
        ax5.set_title('c)', loc='left', fontsize=12)

        # Adjust the layout to prevent overlap
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # Show the plot
        #plt.savefig("uh1_figures/simulated_impulse_estimates.pdf")
        plt.show()

    return


if __name__ == "__main__":
    # simulated_impulse_experiment(show_plot=True)
    ice_excitation_experiment(show_plot=True)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim, lti, lsim
import pickle
import opentorsion as ot

def step_excitation(step_start, step_end, step_value, initial_value=0):
    '''
    A step input function.
    '''

    return lambda t: initial_value + (t >= step_start)*(t < step_end)*step_value

def sinusoidal_excitation(omegas, amplitudes, offset=0, phase=0):
    """
    Sinusoidal excitation function.
    A sum of sine waves.

    Parameters
    omegas : list of floats
        Excitation frequencies (rad/s)
    amplitudes : list of floats
        Excitation amplitudes (Nm)
    offset : float
        DC component of the excitation (Nm)
    phase : float
        Excitation phase

    Returns
    excitation_function : lambda
    """

    return lambda t: offset + sum([amplitudes[i]*0.5*np.sin(omegas[i]*t + phase) for i in range(len(amplitudes))])

def excitation_matrix(t, load, dof):
    '''
    Excitation in matrix form. This function assumes the load is always applied to the last node of the drivetrain.

    Parameters
    t : ndarray
        timesteps
    load : lambda function
        excitation function
    dof : int
        number of degrees of freedom of the drivetrain
    '''
    U = np.zeros(t.shape[0])
    for i in range(1, dof):
        U = np.vstack([U, np.zeros(t.shape[0])])

    U[-1,:] += load(t)

    return U

def drivetrain_3dof():
    '''
    Mechanical drivetrain as an openTorsion assembly instance.

    Returns:

    assembly : opentorsion assembly instance
        A 3-DOF mechanical drivetrain modelled as lumped masses and flexible shafts
        (lumped mass - shaft - lumped mass - shaft - lumped mass).
    '''
    J1 = 0.8 # disk 1 inertia
    J2 = 0.5 # disk 2 inertia
    J3 = 0.7 # disk 3 inertia
    k1 = 1.5e4 # shaft 1 stiffness
    k2 = 1e4 # shaft 2 stiffness

    disks, shafts = [], []
    shafts.append(ot.Shaft(0, 1, None, None, k=k1, I=0))
    shafts.append(ot.Shaft(1, 2, None, None, k=k2, I=0))
    disks.append(ot.Disk(0, I=J1))
    disks.append(ot.Disk(1, I=J2))
    disks.append(ot.Disk(2, I=J3))
    assembly = ot.Assembly(shafts, disk_elements=disks)
    _, f, _ = assembly.modal_analysis()

    return assembly

def state_matrices(assembly):
    '''
    Create state-space matrices A and B of an openTorsion assembly.

    Parameters:

    assembly : openTorsion assembly instance
        Mechanical drivetrain model

    Returns:

    A : numpy.ndarray
        The state matrix
    B : numpy.ndarray
        The input matrix
    '''
    M, K = assembly.M(), assembly.K() # mass and sitffness matrices
    C = assembly.C_modal(M, K, xi=0.02) # modal damping matrix, modal damping coefficient 0.02 used
    Z = np.zeros(M.shape)
    I = np.eye(M.shape[0])
    M_inv = LA.inv(M)

    A = np.vstack([np.hstack([Z, I]), np.hstack([-M_inv @ K, -M_inv @ C])])

    B = np.vstack([Z, M_inv])

    return A, B

def dlti_system(A, B, C, D, dt=1):
    '''
    Returns an instance of a discrete LTI-system using the state-space matrices.
    '''

    return dlti(A, B, C, D, dt=dt)

def lti_system(A, B, C, D):
    '''
    Returns an instance of a continuous LTI-system using the state-space matrices.
    '''

    return lti(A, B, C, D)

def noisy_simulation(time, A, B, C, U):
    '''
    A simulation with process and measurement noise.

    Returns the system output.
    '''
    dt = np.mean(np.diff(time))
    ## add gaussian white noise to measurement and process ##
    R = 1e-3*np.eye(A.shape[0]) # measurement covariance
    Q = 1e-2*np.eye(A.shape[0]) # process covariance
    r = np.random.multivariate_normal(np.zeros(R.shape[0]), R, time.shape[0]).T
    q = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, time.shape[0]).T

    x = np.zeros((A.shape[0], 1))
    y = np.zeros((A.shape[0], 1))

    for i in range(1, time.shape[0]):
        x_new = A @ x + (B @ U[:,i]).reshape(B.shape[0], 1) + q[:,i].reshape(q.shape[0], 1)
        y = np.hstack((y, C @ x_new + r[:,i].reshape(r.shape[0], 1)))
        x = x_new

    return time, y

def simulated_experiment(show_plot=False, pickle_data=False):
    '''
    Simulation of a 3-DOF mechanical drivetrain excited with a sinusoidal excitation.
    '''
    assembly = drivetrain_3dof()
    A, B = state_matrices(assembly)
    C = np.eye(B.shape[0])
    D = np.zeros(B.shape)

    t = np.linspace(0, 100, 101)

    load = step_excitation(20, 21, 100)

    U = excitation_matrix(t, load, dof=assembly.dofs)

    ## continuous time simulation ##
    sys = lti_system(A, B, C, D)
    # tout, yout, xout = lsim(sys, U.T, t)
    tout, yout = noisy_simulation(t, A, B, C, U)

    if show_plot:
        plt.plot(tout, yout[-1,:])
        plt.ylim(0,300)
        plt.show()

    if pickle_data:
        filename = '../data/drivetrain_simulation.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump([t, U, tout, yout], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    simulated_experiment(show_plot=True, pickle_data=False)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim
import pickle
import opentorsion as ot

def excitation(omegas, amplitudes, offset=0, phase=0):
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

def simulated_experiment(show_plot=False, pickle_data=False):
    '''
    Simulation of a 3-DOF mechanical drivetrain excited with a sinusoidal excitation.
    '''
    assembly = drivetrain_3dof()
    A, B = state_matrices(assembly)
    C = np.eye(B.shape[0])
    D = np.zeros(B.shape)

    t = np.linspace(0, 100, 10001)
    dt = np.mean(np.diff(t))

    amplitudes = [300*0.02, 300*0.01] # amplitudes of ~2% and ~1% of 300 Nm
    omegas = [50, 100] # frequencies of 50 and 100 rad/s
    load = excitation(omegas, amplitudes, offset=300)

    U = excitation_matrix(t, load, dof=assembly.dofs)
    plt.plot(t, U[-1,:])
    plt.show()

    sys = dlti_system(A, B, C, D, dt=dt)

    tout, yout, xout = dlsim(sys, U.T, t=t)

    if show_plot:
        plt.plot(tout, yout[:,2])
        plt.show()

    if pickle_data:
        filename = '../data/drivetrain_simulation.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump([tout, yout], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    simulated_experiment(show_plot=True)

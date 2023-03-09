import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy.signal import dlti, dlsim, dimpulse, lti, lsim, impulse
import pickle
import opentorsion as ot

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

def testbench_measured():
    shafts, disks, gears = [], [], []

    shafts = []
    disks = []
    gears = []

    '''Driving motor, motor shaft, aluminium copuling, steel shaft'''
    disks.append(Disk(0, I=6.5e-4))
    shafts.append(Shaft(0, 1, 15, 24))
    shafts.append(Shaft(1, 2, 79.45, 66, idl=20, G=27e9, E=70e9, rho=2710))
    shafts.append(Shaft(2, 3, 6, 16))

    '''Elastomer coupling'''
    shafts.append(Shaft(3, 4, (51.75/3), 32.2, idl=16, G=27e9, E=70e9, rho=2710)) # elastomer coupling hub
    shafts.append(Shaft(4, 5, 0, 0, k=90, I=0)) # elastomer coupling soft part
    shafts.append(Shaft(5, 6, (51.75/3), 32.2, idl=16, G=27e9, E=70e9, rho=2710)) # elastomer coupling middle piece
    shafts.append(Shaft(6, 7, 0, 0, k=90, I=0)) # elastomer coupling soft part
    shafts.append(Shaft(7, 8, (51.75/3), 32.2, idl=16, G=27e9, E=70e9, rho=2710)) # elastomer coupling hub

    '''Shaft, mass, shaft'''
    shafts.append(Shaft(8, 9 , 39, 16))
    disks.append(Disk(9, I=7.7e-3))
    shafts.append(Shaft(9, 10, 39, 16))

    '''Elastomer coupling'''
    shafts.append(Shaft(11, 12, (51.75/3), 32.2, idl=16, G=27e9, E=70e9, rho=2710)) # elastomer coupling hub
    shafts.append(Shaft(12, 13, 0, 0, k=90, I=0)) # elastomer coupling soft part
    shafts.append(Shaft(13, 14, (51.75/3), 32.2, idl=16, G=27e9, E=70e9, rho=2710)) # elastomer coupling middle piece
    shafts.append(Shaft(14, 15, 0, 0, k=90, I=0)) # elastomer coupling soft part
    shafts.append(Shaft(15, 16, (51.75/3), 32.2, idl=16, G=27e9, E=70e9, rho=2710)) # elastomer coupling hub

    '''Shaft, bellow coupling, torque transducer, bellow coupling'''
    shafts.append(Shaft(17, 18, 319, 8)) # long shaft
    shafts.append(Shaft(19, 20, 0, 0, k=15e3, I=2e-5)) # coupling
    shafts.append(Shaft(20, 21, 0, 0, k=5400, I=1.3e-5)) # torque transducer
    shafts.append(Shaft(21, 22, 0, 0, k=15e3, I=2e-5)) # coupling

    '''First gearbox, shafts, couplings'''
    gear1 = Gear(22, I=3.2467e-4, R=1)
    gears.append(gear1) # gear
    gears.append(Gear(23, I=0, R=3, parent=gear1))
    shafts.append(Shaft(23, 24, 0, 0, k=3.1e4, I=1.2e-4)) # coupling
    shafts.append(Shaft(24, 25, 33.5, 16)) # shaft
    shafts.append(Shaft(25, 26, 0, 0, k=3.1e4, I=1.2e-4)) # coupling
    shafts.append(Shaft(26, 27, 62, 25)) # shaft

    '''Second gearbox, shafts, couplings'''
    gear2 = Gear(27, I=7.315601e-5, R=1)
    gears.append(gear2) # shaft & gear
    gears.append(Gear(28, I=1.1265292e-2, R=4, parent=gear2))
    shafts.append(Shaft(28, 29, 105, 24.5)) # shaft
    shafts.append(Shaft(29, 30, 0, 0, k=138e3, I=2.8e-4)) # coupling
    shafts.append(Shaft(30, 31, 0, 0, k=2e4, I=4e-5)) # torque transducer
    shafts.append(Shaft(31, 32, 0, 0, k=138e3, I=2.8e-4)) # coupling
    shafts.append(Shaft(32, 33, 117, 25)) # shaft
    disks.append(Disk(33, I=7.5863e-3)) # mass
    shafts.append(Shaft(33, 34, 77, 25))
    shafts.append(Shaft(34, 35, 0, 0, k=31e3, I=0.12e-3))

    '''Planetary gear, driving motor'''
    gear3 = Gear(21, I=0, R=8) # planetary gear
    gears.append(gear3)
    gears.append(Gear(22, I=1.32e-4, R=1, parent=gear3))
    disks.append(Disk(22, I=6.5e-4)) # load generator

    assembly = Assembly(shafts, disk_elements=disks, gear_elements=gears)
    _, f, _ = assembly.modal_analysis()
    print(f.round(2))

    return assembly

def testbench():
    '''
    Kongsberg testbench openTorsion model.
    '''
    shafts = []
    disks = []
    gears = []

    disks.append(ot.Disk(0, I=6.5e-4))
    shafts.append(ot.Shaft(0, 1, 0, 0, k=1.9039e5 , I=1.4420e-4, c=8.0804)) # driving motor, coupling
    shafts.append(ot.Shaft(1, 2, 0, 0, k=6.9487e3, I=3.7880e-6, c=0.2949)) # shaft
    shafts.append(ot.Shaft(2, 3, 0, 0, k=90, I=3e-6, c=0.2387)) # elastomer coupling hub
    shafts.append(ot.Shaft(3, 4, 0, 0, k=90, I=2e-6, c=0.2387)) # elastomer coupling middle piece
    shafts.append(ot.Shaft(4, 5, 0, 0, k=90, I=0, c=0.2387)) # elastomer coupling hubs & shaft
    disks.append(ot.Disk(5, I=7.8091e-3))
    shafts.append(ot.Shaft(5, 6, 0, 0, k=90, I=2e-6, c=0.2387))# elastomer coupling middle piece
    shafts.append(ot.Shaft(6, 7, 0, 0, k=90, I=0, c=0.0013)) # elastomer coupling hub & shaft
    shafts.append(ot.Shaft(7, 8, 0.342e3, 0.008e3, G=80e9, rho=7800)) # new shaft (shaft & coupling)
    # shafts.append(Shaft(7, 8, 0, 0, k=4.19e4, I=(5.0171e-5+3.1708e-6), c=1.7783)) # old shaft (shaft & coupling)
    shafts.append(ot.Shaft(8, 9, 0, 0, k=5.4e3, I=6.5e-6, c=0.2292)) # torque transducer
    shafts.append(ot.Shaft(9, 10, 0, 0, k=4.19e4, I=5.65e-5, c=1.7783)) # torque transducer & coupling
    shafts.append(ot.Shaft(10, 11, 0, 0, k=1.2192e3, I=4.2685e-6, c=0.5175)) # shaft
    gear1 = ot.Gear(11, I=3.2467e-4, R=1)
    gears.append(gear1) # shaft & gear
    gears.append(ot.Gear(12, I=0, R=3, parent=gear1))
    shafts.append(ot.Shaft(12, 13, 0, 0, k=3.1e4, I=1.2e-4, c=1.3157)) # coupling
    shafts.append(ot.Shaft(13, 14, 0, 0, k=1.1429e3, I=1.1516e-5, c=0.0485)) # shaft
    shafts.append(ot.Shaft(14, 15, 0, 0, k=3.1e4, I=1.3152e-4, c=1.3157)) # shaft & coupling
    shafts.append(ot.Shaft(15, 16, 0, 0, k=1.2192e4, I=4.2685e-6, c=0.5175)) # shaft
    gear2 = ot.Gear(16, I=2.6927e-4, R=1)
    gears.append(gear2) # shaft & gear
    gears.append(ot.Gear(17, I=0, R=4, parent=gear2))
    shafts.append(ot.Shaft(17, 18, 0, 0, k=1.38e5, I=1.8e-4, c=5.8569)) # coupling
    shafts.append(ot.Shaft(18, 19, 0, 0, k=2e4, I=2e-5, c=0.8488)) # torque transducer
    shafts.append(ot.Shaft(19, 20, 0, 0, k=1.38e5, I=2e-4, c=5.8569)) # torque trandsucer & coupling
    shafts.append(ot.Shaft(20, 21, 0, 0, k=1.2192e4, I=4.2685e-6, c=0.5175)) # shaft
    # disks.append(Disk(21, I=7.8e-3)) # shaft, mass, planetary gear & load generator
    disks.append(ot.Disk(21, I=4.9535e-2)) # shaft, mass, planetary gear & load generator

    assembly = ot.Assembly(shafts, disk_elements=disks, gear_elements=gears)
    _, f, _ = assembly.modal_analysis()
    print(f.round(2))

    return assembly

def state_matrices(assembly):
    """
    Create state-space matrices A and B of an openTorsion assembly.

    Parameters:
        assembly : openTorsion assembly instance
            Mechanical drivetrain model

    Returns:
        A : numpy.ndarray
            The state matrix
        B : numpy.ndarray
            The input matrix
    """
    M, C, K = assembly.M(), assembly.C(), assembly.K()  # Mass and stiffness matrices
    Z = np.zeros(M.shape)
    I = np.eye(M.shape[0])
    M_inv = LA.inv(M)

    A = np.vstack([np.hstack([Z, I]), np.hstack([-M_inv @ K, -M_inv @ C])])

    B = np.vstack([Z, M_inv])

    return A, B


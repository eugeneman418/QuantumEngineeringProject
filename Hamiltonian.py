import qutip as qp
import numpy as np
import scipy as sp

def pauliX(i,N):
    return qp.tensor([qp.sigmax() if j == i else qp.identity(2) for j in range(N)])

def pauliY(i,N):
    return qp.tensor([qp.sigmay() if j == i else qp.identity(2) for j in range(N)])

def pauliZ(i,N):
    return qp.tensor([qp.sigmaz() if j == i else qp.identity(2) for j in range(N)])

def PairHamiltonian(N, theta, phi, delta, E, V, Ne):
    Gf, _, _ = sp.constants.physical_constants['Fermi coupling constant'] # TODO: check if the units are consistent
    msw = Gf * Ne/(np.sqrt(2)*V)
    J = np.sqrt(2) * Gf/V * (1 - np.cos(phi)) 
    omega = delta/(4 * E * (N-1))
    
    X1 = pauliX(0,2)
    X2 = pauliX(1,2)
    Y1 = pauliY(0,2)
    Y2 = pauliY(1,2)
    Z1 = pauliZ(0,2)
    Z2 = pauliZ(1,2)
    
    H = omega * np.sin(2 * theta) * (X1 + X2) + (msw - omega * np.cos(2 * theta)) * (Z1 + Z2) + J * (X1 * X2 + Y1 * Y2 + Z1 * Z2)
    return H

def Hamiltonian(N, theta, phi, delta, E, V, Ne):
    """
    t: time
    N: number of neutrinos in system
    theta: mixing angle
    phi: a matrix of propagation angles, where phi[i,j] for i < j is the angle between ith and jth neutrino
    delta: difference of squared mass
    E: energy (in some sources E=2p)
    V: volume of system
    Ne: number of electrons
    """
    Gf, _, _ = sp.constants.physical_constants['Fermi coupling constant'] # TODO: check if the units are consistent
    msw = Gf * Ne/(np.sqrt(2) * V)
    J = np.sqrt(2) * Gf/V
    omega = delta/(4 * E * (N-1))
    
    H = qp.tensor([0 * qp.identity(2) for i in range(N)])
    
    for i in range(N):
        for j in range(i+1,N):
            Xi = pauliX(i,N)
            Yi = pauliY(i,N)
            Zi = pauliZ(i,N)
            Xj = pauliX(j,N)
            Yj = pauliY(j,N)
            Zj = pauliZ(j,N)

            H += omega * np.sin(2 * theta) * Xi + (msw - omega * np.cos(2 * theta)) * Zi
            H += omega * np.sin(2 * theta) * Xj + (msw - omega * np.cos(2 * theta)) * Zj
            H += J * (1 - np.cos(phi[i,j]) ) * (Xi * Xj + Yi * Yj + Zi * Zj)
            
    return H

def OneBodyHamiltonian(N, theta, delta, E, V, Ne):
    Gf, _, _ = sp.constants.physical_constants['Fermi coupling constant'] # TODO: check if the units are consistent
    msw = Gf * Ne/(np.sqrt(2) * V)
    omega = delta/(4 * E)
    
    H = qp.tensor([0 * qp.identity(2) for i in range(N)])
    
    for i in range(N):
            Xi = pauliX(i,N)
            Yi = pauliY(i,N)
            Zi = pauliZ(i,N)

            H += omega * np.sin(2 * theta) * Xi + (msw - omega * np.cos(2 * theta)) * Zi
            
    return H

def TwoBodyHamiltonian(phi, V):
    Gf, _, _ = sp.constants.physical_constants['Fermi coupling constant'] # TODO: check if the units are consistent
    J = np.sqrt(2) * Gf/V * (1 - np.cos(phi) ) 
    
    X1 = pauliX(0,2)
    X2 = pauliX(1,2)
    Y1 = pauliY(0,2)
    Y2 = pauliY(1,2)
    Z1 = pauliZ(0,2)
    Z2 = pauliZ(1,2)
    
    H = J * (X1 * X2 + Y1 * Y2 + Z1 * Z2)
    return H
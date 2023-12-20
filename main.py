import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ReadData
from solver import *
from visualization import *

def main():
    # Input parameters
    SET = Setup()
    E = SET['Young']
    nu = SET['Poisson'] 
    t = SET['Thickness'] #
    Nodals = SET['Nodals']
    EleNo = SET['EleNo']
    DOF = SET['DOF']
    nDOF = SET['nDOF']
    freeDOF = SET['FreeDOF']
    F = SET['Sila']
    
    K = GlobalStiffness(Nodals,EleNo,DOF,nDOF,E,nu,t) 
    K = K.toarray()

    # Deleting rows and collums based on boundary conditions
    Krow = K[freeDOF,:] 
    Kfree = Krow[:,freeDOF]

    u = np.zeros((nDOF,1))  

    # solving system of linear equations
    invK = np.linalg.inv(Kfree)
    u[freeDOF] = invK @ F[freeDOF] # boundary conditions are applied to forces

    # resulting displacements
    uNodal = u.reshape((np.size(Nodals,0),2))

    # magnitude of displacement of node
    uMagnitude = np.zeros((np.size(uNodal,0)))
    for i in range(np.size(uNodal,0)): 
        uMagnitude[i] = np.sqrt(uNodal[i][0]**2 + uNodal[i][1]**2)

    # Stress calculation
    Sigma = Stress(Nodals,EleNo,DOF,E,nu,u)
    principal, vonMisses = Principal_Equivalent(Sigma)

    # Visualization
    visulalizeDisplacement(Nodals,uNodal,EleNo,uMagnitude)
    visualizeStress(Nodals,EleNo, np.concatenate((principal,vonMisses),axis=1))

if __name__ == '__main__':
    Sigma = main()
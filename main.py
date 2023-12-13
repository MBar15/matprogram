import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import ReadData
from solver import *


def main():

    # Read vstupni parametry
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
    K0 =K
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

    scaleValue = 10
    plt.quiver(Nodals[:,0],Nodals[:,1], scaleValue*uNodal[:,0],scaleValue*uNodal[:,1],angles='xy',scale_units='xy',scale=1)
    plt.xlim([-40,600])
    plt.ylim([-40,125])
    plt.grid(0.95)
    
    plt.scatter(Nodals[:,0], Nodals[:,1],alpha=0.5,s=50,c='k')
    barva = np.linalg.norm(uNodal,axis=1)
    plt.scatter(Nodals[:,0]+scaleValue*uNodal[:,0], Nodals[:,1]+scaleValue*uNodal[:,1],s=50,c=barva)
    plt.show()

    Sigma = Stress(Nodals,EleNo,DOF,E,u)

    Principal_Equivalent(Sigma)

    return np.max(abs(Sigma))


if __name__ == '__main__':
    Sigma = main()
        

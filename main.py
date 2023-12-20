import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ReadData
from solver import *


def main():
    print(matplotlib.__version__)
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
    #print(uNodal.max())
    # magnitude of displacement of node
    uMagnitude = np.zeros((np.size(uNodal,0)))
    for i in range(np.size(uNodal,0)): 
        uMagnitude[i] = np.sqrt(uNodal[i][0]**2 + uNodal[i][1]**2)


    Sigma = Stress(Nodals,EleNo,DOF,E,u)

    principal, vonMisses = Principal_Equivalent(Sigma)
    visulalizeDisplacement(Nodals,uNodal,EleNo,uMagnitude)
    visualizeStress(Nodals,EleNo, np.concatenate((principal,vonMisses),axis=1))
    Test(Nodals,EleNo, np.concatenate((principal,vonMisses),axis=1) )
    return np.max(abs(Sigma))


def visulalizeDisplacement(Nodals, uNodal,EleNo,uMagnitude,):
    # arrow visualization
    scaleValue = 10
    plt.quiver(Nodals[:,0],Nodals[:,1], scaleValue*uNodal[:,0],scaleValue*uNodal[:,1],angles='xy',scale_units='xy',scale=1)
    plt.xlim([-40,600])
    plt.ylim([-40,125])
    plt.grid(0.95)
    
    plt.scatter(Nodals[:,0], Nodals[:,1],alpha=0.5,s=50,c='k')
    barva = np.linalg.norm(uNodal,axis=1)
    plt.scatter(Nodals[:,0]+scaleValue*uNodal[:,0], Nodals[:,1]+scaleValue*uNodal[:,1],s=50,c=barva)
    plt.show()

    # Node colors visualization
    x,y = Nodals.T
    triangulation = tri.Triangulation(x,y,EleNo-1)
    # plot the contours
    plt.tricontourf(triangulation, uMagnitude)
    plt.triplot(triangulation,'k-')
    plt.colorbar()
    plt.axis('equal')
    plt.show()


def visualizeStress(Nodals,EleNo,stresses):
    spacing = ([-223.28,-200,-160,-120,-80,-40,0,40,70.46],[-32.41,0,50,100,150,200,250,284.98],[14.09,50,100,150,200,257.09])
    #print([min(stresses[:,0]),max(stresses[:,0])],[min(stresses[:,1]),max(stresses[:,1])],[min(stresses[:,2]),max(stresses[:,2])])
    # Von misses visualization averaged
    for j in range(np.size(stresses,1)):
        fig, ax = plt.subplots()
        stressAtNodes = np.zeros((np.size(Nodals,0)))
        numOfAddedValuesAtNode = np.zeros((np.size(Nodals,0)))

        for i in range(np.size(EleNo,0)):
            stressAtNodes[EleNo[i]-1] += stresses[i,j]
            numOfAddedValuesAtNode[EleNo[i]-1] += 1 
        Average_node_stress = stressAtNodes / numOfAddedValuesAtNode
        #print([min(Average_node_stress),max(Average_node_stress)])
        x,y = Nodals.T
        triangulation = tri.Triangulation(x,y,EleNo-1)
        p =ax.tricontourf(triangulation, Average_node_stress,levels=spacing[j],vmin=min(Average_node_stress),vmax=max(Average_node_stress),cmap='jet') # jiné barvy https://matplotlib.org/stable/gallery/color/colormap_reference.html
        plt.triplot(triangulation,'k-')
        p.set_clim([min(Average_node_stress),max(Average_node_stress)])
        fig.colorbar(p,ax=ax,drawedges=0,ticks=spacing[j])
        plt.xlim([-40,600])
        plt.ylim([-40,125])
        plt.axis('equal')
        plt.show()
        plt.close(fig)



from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib
def Test(Nodals,EleNo,stresses):
    x,y = Nodals.T
    triangulation = tri.Triangulation(x,y,EleNo-1)
    patches = []
    for element in (EleNo-1):    
        triangleXY = Nodals[element]
        triangle = Polygon(triangleXY)
        patches.append(triangle)


    for i in range(np.size(stresses,1)):
        fig, ax = plt.subplots()
        #plt.grid(0.95)
        colors = stresses[:,i]
        p = PatchCollection(patches, cmap='jet') # jiné barvy https://matplotlib.org/stable/gallery/color/colormap_reference.html
        p.set_array(colors)
        p.set_clim(vmin=colors.min(),vmax=colors.max())
        ax.add_collection(p)
        spacing = ([-271.76,-250,-200,-150,-100,-50,0,50,70],[-38.8,0,50,100,150,200,250,284.98],[10.98,50,100,150,200,259.7]) # principal 1, principal 2 , von miss
        fig.colorbar(p,ax=ax,ticks = spacing[i])
        plt.xlim([-40,600])
        plt.ylim([-40,125])
        plt.axis('equal')
        plt.triplot(triangulation,'k-')
        plt.show()
        plt.close(fig)

    


    
#colors = 100 * np.random.rand(len(patches))
#p = PatchCollection(patches, alpha=0.4)
#p.set_array(colors)
#ax.add_collection(p)
#fig.colorbar(p, ax=ax)

if __name__ == '__main__':
    Sigma = main()
        

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def visulalizeDisplacement(Nodals, uNodal,EleNo,uMagnitude,):
    # arrow visualization
    scaleValue = 10
    plt.quiver(Nodals[:,0],Nodals[:,1], scaleValue*uNodal[:,0],scaleValue*uNodal[:,1],angles='xy',scale_units='xy',scale=1)
    plt.xlim([-40,600])
    plt.ylim([-40,125])
    
    plt.scatter(Nodals[:,0], Nodals[:,1],alpha=0.5,s=50,c='k')
    barva = np.linalg.norm(uNodal,axis=1)
    plt.scatter(Nodals[:,0]+scaleValue*uNodal[:,0], Nodals[:,1]+scaleValue*uNodal[:,1],s=50,c=barva)
    plt.title('Posuv vektory')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()

    # Node colors visualization
    x,y = Nodals.T
    triangulation = tri.Triangulation(x,y,EleNo-1)
    # plot the contours
    plt.tricontourf(triangulation, uMagnitude, levels=[0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,uMagnitude.max()])
    plt.triplot(triangulation,'k-')
    plt.colorbar()
    plt.axis('equal')
    plt.title('Posuv')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()

def visualizeStress(Nodals,EleNo,stresses):
    spacing = ([-223.28,-200,-160,-120,-80,-40,0,40,70.46],[-32.41,0,50,100,150,200,250,284.98],[14.09,50,100,150,200,257.09])
    name = ['Hlavní napětí nejmenší (průměrované)','Hlavní napětí největší (průměrované)','Napětí Von-Misses (průměrované)']

    # Stress
    for j in range(np.size(stresses,1)):
        fig, ax = plt.subplots()
        stressAtNodes = np.zeros((np.size(Nodals,0)))
        numOfAddedValuesAtNode = np.zeros((np.size(Nodals,0)))

        for i in range(np.size(EleNo,0)):
            stressAtNodes[EleNo[i]-1] += stresses[i,j]
            numOfAddedValuesAtNode[EleNo[i]-1] += 1 
        Average_node_stress = stressAtNodes / numOfAddedValuesAtNode
        x,y = Nodals.T
        triangulation = tri.Triangulation(x,y,EleNo-1)
        p =ax.tricontourf(triangulation, Average_node_stress,levels=spacing[j],vmin=min(Average_node_stress),vmax=max(Average_node_stress),cmap='viridis') # different colormaps https://matplotlib.org/stable/gallery/color/colormap_reference.html
        plt.triplot(triangulation,'k-')
        p.set_clim([min(Average_node_stress),max(Average_node_stress)])
        fig.colorbar(p,ax=ax,drawedges=0,ticks=spacing[j])
        ax.set_title(name[j])
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        plt.xlim([-40,600])
        plt.ylim([-40,125])
        plt.axis('equal')
        plt.show()

    name = ['Hlavní napětí nejmenší (neprůměrované)','Hlavní napětí největší (neprůměrované)','Napětí Von-Misses (neprůměrované)']
    x,y = Nodals.T
    triangulation = tri.Triangulation(x,y,EleNo-1)
    patches = []
    for element in (EleNo-1):    
        triangleXY = Nodals[element]
        triangle = Polygon(triangleXY)
        patches.append(triangle)

    for i in range(np.size(stresses,1)):
        fig, ax = plt.subplots()
        colors = stresses[:,i]
        p = PatchCollection(patches, cmap='viridis') # different colormaps https://matplotlib.org/stable/gallery/color/colormap_reference.html
        p.set_array(colors)
        p.set_clim(vmin=colors.min(),vmax=colors.max())
        ax.add_collection(p)
        spacing = ([-271.76,-250,-200,-150,-100,-50,0,50,70],[-38.8,0,50,100,150,200,250,284.98],[10.98,50,100,150,200,259.7]) # principal 1, principal 2 , von miss
        fig.colorbar(p,ax=ax,ticks = spacing[i])
        ax.set_title(name[i])
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        plt.xlim([-40,600])
        plt.ylim([-40,125])
        plt.axis('equal')
        plt.triplot(triangulation,'k-')
        plt.show()
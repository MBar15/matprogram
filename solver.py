import numpy as  np
import scipy.sparse as sps
import pandas as pd

def ReadBulkDAT(name="bulk.dat"):
    f = open(name,"r")    
    content = (f.readlines())    
    GRID = []
    EleNoTri = []
    SPC = []
    Force = []
    for i in range(len(content)):
        CheckNOD = content[i].find("GRID")
        CheckTri = content[i].find("CPLSTS3")
        CheckSPC = content[i].find("SPC")
        CheckForce = content[i].find("FORCE")
        if CheckNOD != -1:
            GRID.append(content[i])
        if CheckTri != -1:
            EleNoTri.append(content[i])
        if CheckSPC != -1:
            SPC.append(content[i]) 
        if CheckForce != -1:
            Force.append(content[i])
    GRIDtable=pd.DataFrame([s.split() for s in GRID])    
    Nod = np.float64(GRIDtable.to_numpy()[:,[3,4,5]])
    if np.sum(Nod[:,2]) < 10**-2:
        Nod = Nod[:,[0,1]]
    Eletable=pd.DataFrame([s.split() for s in EleNoTri])    
    EleNo = np.intc(Eletable.to_numpy()[:,3:])
    
    SPCtable= pd.DataFrame([s.split() for s in SPC]) 
    SPCtable.dropna(inplace=True)
    FixNod = np.intc(SPCtable.to_numpy()[:,2])
    
    Forcetable= pd.DataFrame([s.split() for s in Force])
    Forcetable.dropna(inplace=True)
    
    ForceNod = np.intc(Forcetable.to_numpy()[:,2])

    return Nod,EleNo,FixNod,ForceNod



def Setup():
    """setups and accordingly prepares the problem"""
    # Material constants
    t = 10          # thickness mm
    E = 2.1e5       # Young modul MPa
    nu = 0.3        # Poisson ratio
    Force = -10.e3  # Force magnitude applied
    
    Nodals, EleNo, FixDOF, ForceDOF = ReadBulkDAT(name='bulk.dat')
    ET = EleNo.flatten()
    DOF = np.concatenate([[2*ET-1, 2*ET]], axis=0).T
    DOF = DOF.reshape([np.size(EleNo,0), 2*np.size(EleNo,1)]) - 1
 
    DOFu = np.unique(DOF.flatten()) 
    nDOF = np.max(DOFu)+1 

    # Apllying boundary conditions
    clamped_boundary = []
    for i in range(np.size(Nodals,0)):
        if Nodals[i][0] == 0:
            clamped_boundary.append(i)
    clamped_boundary = np.array(clamped_boundary)
    FixDOF = np.concatenate((2*(clamped_boundary+1)-1,2*(clamped_boundary+1)),axis=0)-1    
    FreeDOF = np.setdiff1d(DOFu, FixDOF) 

    F = np.zeros([nDOF,1])  
    Force_boundary = []
    for i in range(np.size(Nodals,0)):
        if Nodals[i][0] == 500:
            Force_boundary.append(i)
    ForceDOF = 2*(np.array(Force_boundary)+1)-1 
    Force_Per_Node = Force/ForceDOF.size
    F[ForceDOF] = Force_Per_Node  * np.ones((ForceDOF.size,1))  
    
    return {'Young':E,'Poisson': nu,'Nodals':Nodals,'EleNo':EleNo,'DOF':DOF,'nDOF':nDOF,'FreeDOF':FreeDOF,'Sila':F,'Thickness':t}

def StiffnessCST(Nodal,E,nu,t):
    D, B, A =CST_matrices(Nodal)
    k = A*t* np.transpose(B) @ D @ B

    return k

def GlobalStiffness(Nodals,EleNo,DOF,nDOF,E,nu,t):

    K = np.zeros((nDOF,nDOF))
    LokalizacniTabulka = np.zeros((0,3))
    for i in np.arange(0,np.size(EleNo,0)):
        ke = StiffnessCST(Nodals[EleNo[i,:]-1,:],E,nu,t)
        Ke = ke.flatten()
        cl = np.kron(np.ones([np.size(DOF[i,:])]), DOF[i,:])
        rw = np.kron(DOF[i,:],np.ones([np.size(DOF[i,:])]))
        RCK = np.stack([rw,cl,Ke],axis=0).T
        LokalizacniTabulka = np.concatenate((LokalizacniTabulka,RCK), axis=0)
        
    K = sps.coo_matrix((LokalizacniTabulka[:,2],(LokalizacniTabulka[:,0],LokalizacniTabulka[:,1])),shape=(nDOF,nDOF))
        
    return (K+K.T)/2


def Stress(Nodals,EleNo,DOF,E,u): #---------------------------------------
    Sigma = np.zeros((np.size(EleNo,0),3))
    for i in np.arange(0,np.size(EleNo,0)):
        uEGlobal = u[DOF[i,:]]
        D, B, A = CST_matrices(Nodals[EleNo[i,:]-1,:])
        Sigma[i,:] =  (D@(B @ uEGlobal)).reshape([1,3])

    return Sigma


def CST_matrices(Nodal):
    E = 2.1e5
    nu = 0.3
    D = E/(1-nu**2) * np.array([
                                [1,nu,0],
                                [nu,1,0],
                                [0,0,(1-nu)/2]
    ])

    Node1 = Nodal[0,:]
    Node2 = Nodal[1,:]
    Node3 = Nodal[2,:]
    b1 = Node2[1]-Node3[1]
    b2 = Node3[1]-Node1[1]
    b3 = Node1[1]-Node2[1]
    c1 = Node3[0]-Node2[0]
    c2 = Node1[0]-Node3[0]
    c3 = Node2[0]-Node1[0]

    A = 1/2*np.linalg.det([
                        [1,Node1[0],Node1[1]],
                        [1,Node2[0],Node2[1]],
                        [1,Node3[0],Node3[1]]
    ])

    B = 1/(2*A) * np.array([
                            [b1,0,b2,0,b3,0],
                            [0,c1,0,c2,0,c3],
                            [c1,b1,c2,b2,c3,b3]
    ])


    return D, B, A

def Principal_Equivalent(Sigma):
    # Principal stress
    res = np.zeros((250,2))
    Tensor = Sigma
    for i in range(np.size(Sigma,0)):
        T = np.array([
            [Tensor[i][0],Tensor[i][2]],
            [Tensor[i][2],Tensor[i][1]],
        ])
        res[i,:] = np.linalg.eigh(T)[0]

    sigmaVM = np.zeros((250,1))
    # Von-Misses
    for i in range(np.size(res,0)):
        sigmaVM[i] = np.sqrt(res[i][0]**2+res[i][1]**2-res[i][0]*res[i][1])

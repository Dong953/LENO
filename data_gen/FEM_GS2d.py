from fenics import *
import numpy as np
from scipy.io import loadmat


A_all = loadmat('./data/data_2d_alpha_2.5.mat')['initialA'] 
S_all = loadmat('./data/data_2d_alpha_2.5.mat')['initialS'] 
DA = 2.5e-4
DS = 5e-4
rho = .04
mu = .065
T = 10.0          
num_steps = 100   
dt = T / num_steps 
h = 1/256
N = A_all.shape[-1]-1
h = 1/N
p0 = Point(0, 0)  
p1 = Point(2*DOLFIN_PI, 2*DOLFIN_PI)  
mesh = RectangleMesh(p0, p1, N, N)
cell = mesh.ufl_cell()
V = FunctionSpace(mesh, 'P', 1)
resultsA = np.zeros([A_all.shape[0],num_steps+1,N+1,N+1])
resultsS = np.zeros([A_all.shape[0],num_steps+1,N+1,N+1])
A_0 = Function(V)
S_0 = Function(V)
idx2d = np.zeros((V.dim(),2),dtype=int)
dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, 2))
x_vals = np.linspace(0, 2*np.pi, N+1) 
y_vals = np.linspace(0, 2*np.pi, N+1) 
for i, (x, y) in enumerate(dof_coordinates):
    ix = np.abs(x_vals - x).argmin()
    iy = np.abs(y_vals - y).argmin()
    idx2d[i,:] = np.array([iy, ix])
for i in range(A_all.shape[0]):
    A_0.vector()[:] = A_all[i,:,:][idx2d[:,0],idx2d[:,1]]
    S_0.vector()[:] = S_all[i,:,:][idx2d[:,0],idx2d[:,1]]
    A = TrialFunction(V)      
    S = TrialFunction(V)            
    A_n = Function(V)        
    S_n = Function(V)
    A_n.assign(A_0)
    S_n.assign(S_0)
    vA = TestFunction(V)
    vS = TestFunction(V)
    
    aA = A / dt * vA * dx + DA*dot(grad(A), grad(vA)) * dx 
    FA = A_n/dt*vA*dx + (S_n*A_n**2-(mu+rho)*A_n)*vA*dx 
    aS = S / dt * vS * dx + DS*dot(grad(S), grad(vS)) * dx 
    FS = S_n/dt*vS*dx + (-S_n*A_n**2+rho*(1.-S_n))*vS*dx 
    t = 0
    resultsA[i,0,idx2d[:,0],idx2d[:,1]] = A_n.vector()[:]
    resultsS[i,0,idx2d[:,0],idx2d[:,1]] = S_n.vector()[:]
    for n in range(num_steps):
        t += dt
        A = Function(V)
        solve(aA == FA, A)
        S = Function(V)
        solve(aS == FS, S)
        A_n.assign(A)
        S_n.assign(S)
        resultsA[i,n+1,idx2d[:,0],idx2d[:,1]] = A.vector()[:]
        resultsS[i,n+1,idx2d[:,0],idx2d[:,1]] = S.vector()[:]
    print(f"Finish {i}",flush=True)
np.savez('./data/data_GS2d.npz', resA=resultsA, resS=resultsS)
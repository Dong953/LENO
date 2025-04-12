from fenics import *
import numpy as np
from scipy.io import loadmat

A_all = loadmat('./data/data_initial_GS.mat')['initialA']  
S_all = loadmat('./data/data_initial_GS.mat')['initialS']  
DA = 2.5e-4
DS = 5e-4
rho = .04
mu = .065
T = .6          
num_steps = 60   
dt = T / num_steps 
h = 1/1024
N = int(1/h)
mesh = IntervalMesh(N,0,2*np.pi)
cell = mesh.ufl_cell()
VA = FiniteElement("CG", cell, 1)
VS = FiniteElement("CG", cell, 1)
Ve = MixedElement((VA,VS))
V = FunctionSpace(mesh,Ve)
resultsA = np.zeros([A_all.shape[0],num_steps+1,N+1])
resultsS = np.zeros([A_all.shape[0],num_steps+1,N+1])
u_0 = Function(V)
for i in range(A_all.shape[0]):
    tt = np.zeros(2*(N+1))
    tt[::2] = np.flip(A_all[i, :]).astype(u_0.vector()[:].dtype)
    tt[1::2] = np.flip(S_all[i, :]).astype(u_0.vector()[:].dtype)
    u_0.vector()[:] = tt
    (A,S) = TrialFunction(V)            
    u_n = Function(V)          
    (A_n, S_n) = split(u_n)
    u_n.assign(u_0)
    (vA,vS) = TestFunction(V)
    
   
    a = A / dt * vA * dx +  S / dt * vS * dx + DA*dot(grad(A), grad(vA)) * dx + DS*dot(grad(S), grad(vS)) * dx 
    F = A_n/dt*vA*dx + S_n/dt*vS*dx + (S_n*A_n**2-(mu+rho)*A_n)*vA*dx + (-S_n*A_n**2+rho*(1.-S_n))*vS*dx
    
    t = 0
    resultsA[i,0,:] = np.array(np.flip(u_n.vector()[::2]))
    resultsS[i,0,:] = np.array(np.flip(u_n.vector()[1::2]))
    for n in range(num_steps):
        t += dt
        u = Function(V)
        solve(a == F, u)
        u_n.assign(u)

        resultsA[i,n+1,:] = np.array(np.flip(u.vector()[::2]))
        resultsS[i,n+1,:] = np.array(np.flip(u.vector()[1::2]))
np.save('./data/data_GS.npy', results)
   
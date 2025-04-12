from fenics import *
import numpy as np
from scipy.io import loadmat

u0_all = loadmat('./data/data_neu.mat')['input']  # 将 'filename.mat' 替换为你的文件名
epsilon = .1
T = 3         
num_steps = 60    
dt = T / num_steps
h = 1/1024
N = int(1/h)
mesh = IntervalMesh(N,0,2*np.pi)
=V = FunctionSpace(mesh, 'P', 1)

u_0 = Function(V)
results = np.zeros([u0_all.shape[0],num_steps+1,N+1])
noise = Function(V)

for i in range(u0_all.shape[0]):
    u_0.vector()[:] = np.flip(u0_all[i, :]).astype(u_0.vector()[:].dtype)
    
    u = TrialFunction(V)          
    u_n = Function(V)        
    u_n.assign(u_0)            
    
    v = TestFunction(V)
    
    a = u / dt * v * dx + epsilon**2*dot(grad(u), grad(v)) * dx 
    F = u_n/dt*v*dx + (u_n-u_n**3)*v*dx
    t = 0
    results[i,0,:] = np.array(np.flip(u_n.vector()[:]))
    
    for n in range(num_steps):
        t += dt
        u = Function(V)
        solve(a == F, u)
        u_n.assign(u)
        results[i,n+1,:] = np.array(np.flip(u_n.vector()[:]))
np.save('./data/data_ac_neu.npy', results)
  
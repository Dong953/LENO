from fenics import *
import numpy as np
from scipy.io import loadmat

u0_all = loadmat('./data/data.mat')['input']  
u0_all *= 100
T = 1.0           
num_steps = 100    
dt = T / num_steps 
kappa = 1.0      
lam = 1.0     
h = 1/1024
N = int(1/h)
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, 'P', 1)
u_D = Constant(0.)
bc = [DirichletBC(V, u_D, "on_boundary")]
x = np.linspace(0,1,N+1)
u_0 = Function(V)
results = np.zeros([u0_all.shape[0],num_steps+1,N+1])
noise = Function(V)
for i in range(u0_all.shape[0]):
    u_0.vector()[:] = np.flip(u0_all[i, :]).astype(u_0.vector()[:].dtype)
    
    u = Function(V)
    u = TrialFunction(V)         
    u_n = Function(V)         
    
    v = TestFunction(V)
    f = lam * u_n * (kappa - u_n)  
    
    a = u / dt * v * dx + dot(grad(u), grad(v)) * dx 
    F = f * v * dx + u_n/dt*v*dx 
    t = 0
    results[i,0,:] = np.array(np.flip(u_n.vector()[:]))
    
    for n in range(num_steps):
        t += dt
        u = Function(V)
        solve(a == F, u, bc)
        u_n.assign(u)
        results[i,n+1,:] = np.array(np.flip(u_n.vector()[:]))
np.save('./data/data_kpp'+'.npy', results)

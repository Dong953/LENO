from fenics import *
import numpy as np
from scipy.io import loadmat
import os
A_all = loadmat('./data/data_2d_dirichlet.mat')['initialA'] 
A_all *= 100
T = .01        
num_steps = 100   
dt = T / num_steps 
N = A_all.shape[-1]-1
h = 1/N
p0 = Point(-8, -8)  
p1 = Point(8, 8)  
mesh = RectangleMesh(p0, p1, N, N)
cell = mesh.ufl_cell()
V = FunctionSpace(mesh, 'P', 1)
V_expr = Expression("100 * (pow(sin(DOLFIN_PI*x[0]/4), 2) + pow(sin(DOLFIN_PI*x[1]/4), 2) ) + pow(x[0], 2) + pow(x[1], 2) ",degree=2)
alpha = Constant(1600.)
lambda_ = Constant(15.849750955787211)
results = np.zeros([A_all.shape[0],num_steps+1,N+1,N+1])
A_0 = Function(V)
idx2d = np.zeros((V.dim(),2),dtype=int)
dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, 2))
x_vals = np.linspace(-8, 8, N+1) 
y_vals = np.linspace(-8, 8, N+1) 
for i, (x, y) in enumerate(dof_coordinates):
    ix = np.abs(x_vals - x).argmin()
    iy = np.abs(y_vals - y).argmin()
    idx2d[i,:] = np.array([iy, ix])
u_D = Constant(0.0)
bc = DirichletBC(V, u_D, "on_boundary")
for i in range(1):
    A_0.vector()[:] = A_all[i,:,:][idx2d[:,0],idx2d[:,1]]
    A = TrialFunction(V)      
    A_n = Function(V)        
    A_n.assign(A_0)
    vA = TestFunction(V)
    aA = A / dt * vA * dx + dot(grad(A), grad(vA)) * dx 
    FA = A_n/dt*vA*dx - alpha*abs(A_n)**2*A_n*vA*dx - V_expr*A_n*vA*dx + lambda_*A_n*vA*dx
    t = 0
    results[i,0,idx2d[:,0],idx2d[:,1]] = A_n.vector()[:]
    for n in range(num_steps):
        t += dt
        A = Function(V)
        solve(aA == FA, A, bc)
        A_n.assign(A)
        results[i,n+1,idx2d[:,0],idx2d[:,1]] = A.vector()[:]
    print(f"Finish {i}",flush=True)
print(np.isnan(results).any())
np.savez('./data/data_Schrodinger_2d.npz', res=results)




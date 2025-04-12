import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mshr import *
from dolfin import *
from slepc4py import SLEPc
import scipy.io as sio
import os

CN_data = torch.zeros(79,3,48,48,48)
for i in range(79):
    m = i+1
    fie_name = str(m)
    file_path1 = f'CN/{m}/T1_48_bl.nii.gz'
    img1 = nib.load(file_path1)
    data1 = img1.get_fdata()
    data_torch = torch.Tensor(data1)
    CN_data[i,0,...] = data_torch
for i in range(79):
    m = i+1
    fie_name = str(m)
    file_path1 = f'CN/{m}/T1_48_m06.nii.gz'
    img1 = nib.load(file_path1)
    data1 = img1.get_fdata()
    data_torch = torch.Tensor(data1)
    CN_data[i,1,...] = data_torch
for i in range(79):
    m = i+1
    fie_name = str(m)
    file_path1 = f'CN/{m}/T1_48_m12.nii.gz'
    img1 = nib.load(file_path1)
    data1 = img1.get_fdata()
    data_torch = torch.Tensor(data1)
    CN_data[i,2,...] = data_torch
CN_dataz = CN_data[...,24]
data_max = CN_dataz.max()
res = CN_dataz.cpu().numpy()
mask = res[0,0,...]>0
np.savez('../data/patient_all.npz',res=res)
def get_eigenfunction(u,mask):
    t = np.zeros_like(mask)*0.
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            p = Point(j/50,i/50)
            try:
                m = u(p)                
            except:
                m = 0.
            t[i,j] = m
    return t
contours = measure.find_contours(mask, level=0.5)
largest_contour = max(contours, key=len)
polygon_points = [Point(pt[1]/50, pt[0]/50) for pt in largest_contour]
domain = Polygon(polygon_points)
mesh = generate_mesh(domain, 100)
V = FunctionSpace(mesh,'P',1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx + 1e-4*u*v*dx
m = u*v*dx
A = PETScMatrix()
M = PETScMatrix()
assemble(a, tensor=A)
assemble(m, tensor=M)
eigensolver = SLEPcEigenSolver(A, M)
eigensolver.parameters["spectrum"] = "smallest magnitude"
eigensolver.parameters["tolerance"] = 1e-8 
eigensolver.parameters["maximum_iterations"] = 1000 
number_of_eigenvalues = 1024
eigensolver.solve(number_of_eigenvalues)
eigenv = np.zeros(number_of_eigenvalues)
eigenf = np.zeros((number_of_eigenvalues,mask.shape[0],mask.shape[1]))
for i in range(number_of_eigenvalues):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    print("Eigenvalue %d: %g" % (i+1, r),flush=True)
    u_eig = Function(V)
    u_eig.vector()[:] = rx
    t = assemble(u_eig*u_eig*dx)**.5
    eigenv[i] = r
    u_eig.vector()[:] = u_eig.vector()[:]/t
    eigenf[i,...] = get_eigenfunction(u_eig,mask)
eigenv = eigenv-1e-4
eigenv[0] = 0.
u = Function(V)
u.interpolate(Constant(1.))
mask_eval = get_eigenfunction(u,mask)
np.savez(f'./data/eigen_brain_{number_of_eigenvalues}.npz',eigenf=eigenf,eigenv=eigenv,mask_eval=mask_eval)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
data = np.load('../data/patient_all.npz')['res']
device = 'cuda:1'
data_eigen = np.load('../data/eigen_brain_1024.npz')
data = torch.tensor(data).to(device)
eigenf = data_eigen['eigenf']
eigen = data_eigen['eigenv']
mask_eval = data_eigen['mask_eval']
p = 2048
eigenf = torch.tensor(eigenf).to(device)
eigen = torch.tensor(eigen).to(device)
mask_eval = torch.tensor(mask_eval).to(device)
data = data*mask_eval.unsqueeze(0).unsqueeze(0)
betaA = torch.zeros(data.shape[0],3,p,dtype=torch.float64).to(device)
for j in range(p):
    basis = eigenf[j,...]
    bb = basis**2
    Ac = data*basis.unsqueeze(0).unsqueeze(0)
    Ac = .5*(Ac[...,:-1]+Ac[...,1:])
    Ac = .5*(Ac[...,:-1,:]+Ac[...,1:,:])
    bb = .5*(bb[...,:-1]+bb[...,1:])
    bb = .5*(bb[...,:-1,:]+bb[...,1:,:])
    betaA[...,j] = torch.mean(Ac,(-2,-1))/torch.mean(bb)
betaA = betaA.cpu().numpy()
np.save('../data/dat_coe.npz',betaA)

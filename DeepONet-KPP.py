import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utilities3 import *
from timeit import default_timer as timer
import sys
device = 'cuda:2'
T = 1.0
steps = 100
dt = T / steps
sol = np.load('./data/data_kpp.npy')
num_points = sol.shape[-1]
x = np.linspace(0, 1, num_points)
p = 64
M = 1000
ntrain = 100
beta = np.zeros([sol.shape[0],sol.shape[1],p])
for i in range(p):
    basis = sol*np.sin((i+1)*np.pi*x)
    beta[:,:,i] = np.mean(basis[...,:-1]+basis[...,1:],-1)
target = beta
myall = torch.tensor(target[:ntrain,:,:]).to(device)
initial = torch.tensor(target[:ntrain,0,:]).to(device)
target = torch.tensor(target[:ntrain,1:,:]).to(device)
eigen = (torch.linspace(1,p,p)*torch.pi).to(device)**2
ini = torch.tensor(sol[:ntrain,0,:]).to(device)
yall = torch.tensor(sol[:ntrain,:,:]).to(device)
num_steps = 20
class NonlinearNet(nn.Module):
    def __init__(self, n, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, p)
        )
        self.net1 = nn.Sequential(
            nn.Linear(1,M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M,p)
        )
        self.p = p
    def forward(self, u, x):
        alpha = self.net(u)
        beta = self.net1(x)
        c = torch.einsum('...i,ji->...j',alpha,beta)
        return c
    def forward2(self, u, x):
        alpha = self.net(u)
        beta = self.net1(x)
        c = torch.einsum('...i,ji->...j',alpha,beta)
        beta = torch.zeros(*u.shape[:-1],self.p).to(device)
        for i in range(self.p):
            basis = c*torch.sin((i+1)*np.pi*x.squeeze(-1))
            beta[...,i] = torch.mean(basis[...,1:]+basis[...,:-1],-1)
        return beta
epochs = 5000
lr = .5*1e-3
net = NonlinearNet(sol.shape[-1],M,p).to(device)
net = net.double()
print(count_params(net))
optimizer = optim.Adam(net.parameters(), lr=lr)
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_err = []
x = torch.tensor(x).to(device)
x1 = x.unsqueeze(-1)
f_co = (myall[:,1:num_steps+1,...]-myall[:,:num_steps,...])/dt + myall[:,1:num_steps+1,...]*eigen
fA = torch.zeros([ntrain,num_steps,num_points]).to(device)
for j in range(p):
    fA += f_co[...,j].unsqueeze(-1)*((torch.sin((j+1)*torch.pi*x)).unsqueeze(0).unsqueeze(0))
for i in range(epochs):
    t1 = timer()

    u0 = initial
    uini = ini
    loss = 0
    for step in range(num_steps):
        u = (u0 + dt*net.forward2(uini,x1))/(1+dt*eigen)
        loss += torch.mean(torch.norm(u-target[:,step,:],2,-1)/torch.norm(target[:,step,:],2,-1))/num_steps
        Non = net(yall[:,step,:],x1)
        ffA = fA[:,step,...]
        loss += torch.mean(torch.norm(ffA - Non,2,-1)/torch.norm(ffA,2,-1))/num_steps
        u0 = u
        output = torch.zeros([ntrain,num_points]).double().to(device)
        for j in range(p):
                output += u0[...,j].unsqueeze(-1)*((torch.sin((j+1)*torch.pi*x)).unsqueeze(0))     
        uini = output
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    t2 = timer()
    if (i+1) % 5 == 0:
        print(f"Step {i+1}, Loss: {loss.item()}")

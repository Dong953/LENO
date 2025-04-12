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

device = 'cuda:3'
dt = 5/100
num_steps = 20
sol = np.load('data/data_ac_neu.npy')
epsilon = .1
num_points = sol.shape[-1]
x = np.linspace(0, 2*np.pi, num_points)
p = 64
M = 1000
ntrain = sol.shape[0]
beta = np.zeros([sol.shape[0],sol.shape[1],p])
nor = 2*np.ones([p])
nor[0] = 1.
for i in range(p):
    basis = np.cos(i*x/2)
    a = sol*basis
    beta[:,:,i] = np.mean((a[...,:-1]+a[...,1:])/2,-1)*nor[i]
target = beta
myall = torch.tensor(target[:ntrain,:,:]).to(device)
initial = torch.tensor(target[:ntrain,0,:]).to(device)
target = torch.tensor(target[:ntrain,1:,:]).to(device)
eigen = (torch.linspace(0,p-1,p)).to(device)**2/4
class NonlinearNet(nn.Module):
    def __init__(self, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(p, M),
            nn.ReLU(),
            nn.Linear(M, p)
        )
    def forward(self, u):
        return self.net(u)

epochs = 5000
lr = 1e-3
net = NonlinearNet(M,p).to(device)
net = net.double()
print(count_params(net))
optimizer = optim.Adam(net.parameters(), lr=lr)
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = nn.MSELoss()
train_err = []
for i in range(epochs):
    u0 = initial
    loss = 0
    for step in range(num_steps):
        u = (u0 + dt*net(u0))/(1+epsilon**2*dt*eigen)
        loss += torch.mean(torch.norm(u-target[:,step,:],2,-1)/torch.norm(target[:,step,:],2,-1))/num_steps
        ff = (myall[:,step+1,:]-myall[:,step,:])/dt + epsilon**2*myall[:,step+1,:]*eigen
        loss += torch.mean(torch.norm(ff - net(myall[:,step,:]),2,-1))/num_steps
        u0 = u
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    if (i+1) % 50 == 0:
        print(f"Step {i+1}, Loss: {loss.item()}")
with torch.no_grad():
    sol = torch.tensor(sol).to(device)
    u0 = initial
    u_0 = initial
    xx = torch.linspace(0,2*torch.pi,num_points).to(device).double()
    loss = 0
    loss1 = 0
    net.eval()
    for step in range(num_steps):
        u = (u0 + dt*net(u0))/(1+epsilon**2*dt*eigen)
        uu = torch.zeros([ntrain,num_points]).double().to(device)
        for j in range(p):
            uu += u[:,j].unsqueeze(-1)*torch.cos((j)*xx/2).unsqueeze(0)
        loss += torch.mean(torch.norm((uu-sol[:ntrain,step+1,:]),2,-1)/torch.norm(sol[:ntrain,step+1,:],2,-1))/num_steps
        loss1 += torch.mean(torch.norm(((myall[:,step+1,:]-myall[:,step,:])/dt + epsilon**2*myall[:,step+1,:]*eigen - net(myall[:,step,:])),2,-1))/num_steps
        u_0 = u0
        u0 = u
    tt = torch.tensor((1.-sol[:,:,...]**2)*sol[:,:,...]).to(device)
    bb = torch.tensor(beta).to(device)
    u = net(bb[:,:,...])
    xx = torch.linspace(0,2*torch.pi,num_points).to(device).double()
    output = torch.zeros_like(tt)
    for j in range(p):
        output += u[:,:,j].unsqueeze(-1)*torch.cos((j)*xx/2).unsqueeze(0)
    output = output[:ntrain,:num_steps,:]
    tt = tt[:ntrain,:num_steps,:]
    loss3 = torch.mean(torch.norm(output-tt,2,-1)/torch.norm(tt,2,-1))
    print("L2 loss: {:2.2e}, Residual loss: {:2.2e}, Nonlinear loss Relative: {:2.2e}".format(loss,loss1,loss3))

import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = 'cuda:6'

dt = .05
num_steps = 20
sol = np.load('data/data_ac_neu.npy')
epsilon = .1
num_points = sol.shape[-1]
x = np.linspace(0, 2*np.pi, num_points)
p = 64
M = 1000
ntrain = 110
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
ini = torch.tensor(sol[:ntrain,0,:]).to(device)
yall = torch.tensor(sol[:ntrain,:,:]).to(device)

class NonlinearNet(nn.Module):
    def __init__(self, n, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, M),
            nn.ReLU(),
            nn.Linear(M, p)
        )
        self.net1 = nn.Sequential(
            nn.Linear(1,M),
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
        nor = 2*torch.ones([p]).double().to(device)
        nor[0] = 1.
        for i in range(p):
            basis = torch.cos(i*x.squeeze(-1)/2)
            a = c*basis
            beta[...,i] = torch.mean((a[...,:-1]+a[...,1:])/2,-1)*nor[i]
        return beta
epochs = 5000
lr = 1e-4
net = NonlinearNet(sol.shape[-1],M,p).to(device)
net = net.double()
optimizer = optim.Adam(net.parameters(), lr=lr)
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_err = []
x = torch.tensor(x).to(device)
x1 = x.unsqueeze(-1)
f_co = (myall[:,1:,...]-myall[:,:-1,...])/dt + epsilon**2*myall[:,1:,...]*eigen
fA = torch.zeros([ntrain,steps,num_points]).to(device)
for j in range(p):
    fA += f_co[...,j].unsqueeze(-1)*((torch.cos((j)*x/2)).unsqueeze(0).unsqueeze(0))
for i in range(epochs):
    u0 = initial
    uini = ini
    loss = 0
    for step in range(num_steps):
        u = (u0 + dt*net.forward2(uini,x1))/(1+epsilon**2*dt*eigen)
        loss += torch.mean(torch.norm(u-target[:,step,:],2,-1)/torch.norm(target[:,step,:],2,-1))/num_steps
        Non = net(yall[:,step,:],x1)
        ffA = fA[:,step,...]
        loss += torch.mean(torch.norm(ffA - Non,2,-1)/torch.norm(ffA,2,-1))/num_steps
        u0 = u
        output = torch.zeros([ntrain,num_points]).double().to(device)
        for j in range(p):
                output += u0[...,j].unsqueeze(-1)*((torch.cos((j)*x/2)).unsqueeze(0))     
        uini = output
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    if (i+1) % 5 == 0:
        print(f"Step {i+1}, Loss: {loss.item()}",flush=True)

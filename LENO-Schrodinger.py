#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:01:04 2025

@author: wangjindong
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:4'
dt = 1e-4
num_steps = 20
data = np.load('./data/data_Schrodinger_2d.npz')
solA = data['res']
ntrain = solA.shape[0]
solA = solA[:ntrain,...]
DA = 1.
num_points = solA.shape[-1]
x = np.linspace(-8, 8, num_points)
X,Y = np.meshgrid(x,x)
p = 32
M = 2000
betaA = np.zeros([solA.shape[0],solA.shape[1],4,p,p])
nor = 2.
eigen = np.zeros([4,p,p])
betaA = torch.tensor(betaA).to(device)
solA = torch.tensor(solA).to(device)
X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)
for i in range(p):
    for j in range(p):
        basis1 = torch.sin((i+1)*np.pi*X/8)*torch.sin((j+1)*np.pi*Y/8)
        basis2 = torch.cos((i+1/2)*np.pi*X/8)*torch.cos((j+1/2)*np.pi*Y/8)
        basis3 = torch.sin((i+1)*np.pi*X/8)*torch.cos((j+1/2)*np.pi*Y/8)
        basis4 = torch.cos((i+1/2)*np.pi*X/8)*torch.sin((j+1)*np.pi*Y/8)
        Ac = solA*basis1*4
        Ac = .5*(Ac[...,:-1]+Ac[...,1:])
        Ac = .5*(Ac[...,:-1,:]+Ac[...,1:,:])
        betaA[...,0,i,j] = torch.mean(Ac,(2,3))
        eigen[0,i,j] = ((i+1)**2+(j+1)**2)/64*np.pi**2
        Ac = solA*basis2*4
        Ac = .5*(Ac[...,:-1]+Ac[...,1:])
        Ac = .5*(Ac[...,:-1,:]+Ac[...,1:,:])
        betaA[...,1,i,j] = torch.mean(Ac,(2,3))
        eigen[1,i,j] = ((i+1/2)**2+(j+1/2)**2)/64*np.pi**2
        Ac = solA*basis3*4
        Ac = .5*(Ac[...,:-1]+Ac[...,1:])
        Ac = .5*(Ac[...,:-1,:]+Ac[...,1:,:])
        betaA[...,2,i,j] = torch.mean(Ac,(2,3))
        eigen[2,i,j] = ((i+1)**2+(j+1/2)**2)/64*np.pi**2
        Ac = solA*basis4*4
        Ac = .5*(Ac[...,:-1]+Ac[...,1:])
        Ac = .5*(Ac[...,:-1,:]+Ac[...,1:,:])
        betaA[...,3,i,j] = torch.mean(Ac,(2,3))
        eigen[3,i,j] = ((i+1/2)**2+(j+1)**2)/64*np.pi**2
targetA = betaA
myallA = targetA[:ntrain,...]
initialA = targetA[:ntrain,0,...]
targetA = targetA[:ntrain,1:,...]
eigen = torch.tensor(eigen).to(device)
class NonlinearNet(nn.Module):
    def __init__(self, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4*p*p, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, 4*p*p)
        )
        self.p = p
    def forward(self, u):
        return self.net(u.view(*u.shape[:-3],-1)).view(*u.shape[:-3],4,self.p,self.p)
epochs = 5000
lr = 1e-2
net = NonlinearNet(M,p).to(device)
net = net.double()
optimizer = optim.Adam(net.parameters(), lr=lr)
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_err = []
for i in range(epochs):
    u0A = initialA
    loss = 0
    for step in range(num_steps):
        fcoe = net(u0A)
        uA = (u0A + dt*fcoe)/(1+DA*dt*eigen)
        loss += torch.mean(torch.norm(uA-targetA[:,step,...],2,(-1,-2,-3))/(torch.norm(targetA[:,step,...],2,(-1,-2,-3))))/num_steps
        ffA = (myallA[:,step+1,...]-myallA[:,step,...])/dt + DA*myallA[:,step+1,...]*eigen
        Non = net(myallA[:,step,...])
        loss += torch.mean(torch.norm(ffA - Non,2,(-1,-2,-3))/(torch.norm(ffA,2,(-1,-2,-3))))/num_steps
        u0A = uA
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    if (i+1) % 50 == 0:
        print(f"Step {i+1}, Loss: {loss.item()}")
solA = torch.tensor(solA).to(device)
net.eval()
with torch.no_grad():
    u0A = initialA
    lossA = 0
    loss1A = 0
    for step in range(num_steps):
        fcoe = net(u0A)
        uA = (u0A + dt*fcoe)/(1+DA*dt*eigen)
        outA = torch.zeros(solA.shape[0],solA.shape[-1],solA.shape[-1]).to(device)
        for i in range(p):
           for j in range(p):
               outA += uA[:,1,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i+1/2)*torch.pi*X/8)*torch.cos((j+1/2)*torch.pi*Y/8)).unsqueeze(0))
               outA += uA[:,0,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.sin((i+1)*torch.pi*X/8)*torch.sin((j+1)*torch.pi*Y/8)).unsqueeze(0))
               outA += uA[:,3,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i+1/2)*torch.pi*X/8)*torch.sin((j+1)*torch.pi*Y/8)).unsqueeze(0))
               outA += uA[:,2,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.sin((i+1)*torch.pi*X/8)*torch.cos((j+1/2)*torch.pi*Y/8)).unsqueeze(0))
        lossA += torch.mean(torch.norm(outA-solA[:,step+1,...],2,(-1,-2))/(torch.norm(solA[:,step+1,...],2,(-1,-2))))/num_steps
        ffA = (myallA[:,step+1,...]-myallA[:,step,...])/dt + DA*myallA[:,step+1,...]*eigen
        Non = net(myallA[:,step,...])
        loss1A += torch.mean(torch.norm(ffA - Non,2,(-1,-2,-3))/(torch.norm(ffA,2,(-1,-2,-3))))/num_steps
        u0A = uA
    A = solA
    alpha = 1600
    V = 100*(torch.sin(np.pi*X/4)**2+torch.sin(np.pi*Y/4)**2)+X**2+Y**2
    lambda_ = 15.849750955787211
    fA = lambda_*A-alpha*A**3-V*A
    u = net(betaA)
    outputA = torch.zeros_like(fA)
    for i in range(p):
       for j in range(p):
           outputA += u[:,:,1,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i+1/2)*torch.pi*X/8)*torch.cos((j+1/2)*torch.pi*Y/8)).unsqueeze(0))
           outputA += u[:,:,0,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.sin((i+1)*torch.pi*X/8)*torch.sin((j+1)*torch.pi*Y/8)).unsqueeze(0))
           outputA += u[:,:,3,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i+1/2)*torch.pi*X/8)*torch.sin((j+1)*torch.pi*Y/8)).unsqueeze(0))
           outputA += u[:,:,2,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.sin((i+1)*torch.pi*X/8)*torch.cos((j+1/2)*torch.pi*Y/8)).unsqueeze(0))
    loss2A = torch.mean(torch.norm(outputA[:ntrain,:num_steps+1,...]-fA[:ntrain,:num_steps+1,...],2,(-2,-1))/torch.norm(fA[:ntrain,:num_steps+1,...],2,(-2,-1)))
    print("A: L2 loss: {:2.2e}, Residual loss: {:2.2e},  Nonlinear loss Relative: {:2.2e}".format(lossA,loss1A,loss2A))

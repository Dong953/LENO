import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
device = 'cuda:0'
num_steps = 2
class NonlinearNet(nn.Module):
    def __init__(self, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(p*p, M),
            nn.ReLU(),
            nn.Linear(M, p*p)
        )
        self.p = p
    def forward(self, u):
        return self.net(u.view(*u.shape[:-2],-1)).view(*u.shape[:-2],self.p,self.p)
dt = 1
for NN in range(10,21):
    data = np.load('./data/patient_single_all.npz')
    solA = data['a_trunc']
    ntrain = 1
    solA = solA[NN,:ntrain,...]
    DA = nn.Parameter(torch.tensor(0.).double().to(device))
    num_points = solA.shape[-1]
    x = np.linspace(0, 1, num_points)
    X,Y = np.meshgrid(x,x)
    p = 16
    M = 16
    betaA = np.zeros([solA.shape[0],solA.shape[1],p,p])
    nor = 2.*np.ones([p])
    nor[0] = 1.
    eigen = np.zeros([p,p])
    betaA = torch.tensor(betaA).to(device)
    solA = torch.tensor(solA).to(device)
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    for i in range(p):
        for j in range(p):
            basis = torch.cos(np.pi*i*X)*torch.cos(np.pi*j*Y)
            Ac = solA*basis*nor[i]*nor[j]
            Ac = .5*(Ac[...,:-1]+Ac[...,1:])
            Ac = .5*(Ac[...,:-1,:]+Ac[...,1:,:])
            betaA[...,i,j] = torch.mean(Ac,(2,3))
            eigen[i,j] = ((i)**2+(j)**2)*np.pi**2
    targetA = betaA
    myallA = targetA[:ntrain,...]
    initialA = targetA[:ntrain,0,...]
    targetA = targetA[:ntrain,1:,...]
    eigen = torch.tensor(eigen).to(device)
    net = NonlinearNet(M,p).to(device)
    net = net.double()
    net = torch.load('ckpt/npde_ct_p_16_M_16.pth',map_location=device)
    DA = torch.load('ckpt/npde_ct_DA_p_16_M_16.pth',map_location=device).data
    Dt = nn.Parameter(torch.tensor(1.).double().to(device))
    for i, layer in enumerate(net.net):
        if isinstance(layer, nn.Linear) and i != len(net.net) - 1:  # 跳过最后一层
            for param in layer.parameters():
                param.requires_grad = False
    epochs = 5000
    lr = 1e-3
    step_size = 1000
    optimizer = torch.optim.Adam(
        [
            {'params': [Dt]},  # 单独优化额外参数 Dt
            {'params': filter(lambda p: p.requires_grad, net.net.parameters())}  # 优化 net 中需要梯度的参数
        ],
        lr=lr,
        weight_decay=1e-8
    )
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_err = []
    for i in range(epochs):
        u0A = initialA
        loss = 0
        for step in range(num_steps):
            fcoe = net(u0A)
            uA = (u0A + Dt*dt*fcoe)/(1+Dt*DA*dt*eigen)
            loss += torch.mean(torch.norm(uA-targetA[:,step,...],2,(-1,-2))/(torch.norm(targetA[:,step,...],2,(-1,-2))))/num_steps
            ffA = (myallA[:,step+1,...]-myallA[:,step,...])/(Dt*dt) + DA*myallA[:,step+1,...]*eigen
            Non = net(myallA[:,step,...])
            loss += torch.mean(torch.norm(ffA - Non,2,(-1,-2))/(torch.norm(ffA,2,(-1,-2))))/num_steps
            u0A = uA
        
            # Backpropagate the error and update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_err.append(loss.item())
        # Print loss every 10 steps
        # if (i+1) % 50 == 0:
        #     print(f"Step {i+1}, Loss: {loss.item()}")
    net.eval()
    num_steps = 2
    u0A = initialA
    lossA = 0
    loss0A = 0
    loss1A = 0
    for step in range(num_steps):
        fcoe = net(u0A)
        uA = (u0A + dt*fcoe)/(1+DA*dt*eigen)
        outA = torch.zeros(solA.shape[0],solA.shape[-1],solA.shape[-1]).to(device)
        for i in range(p):
            for j in range(p):
                outA += uA[:,i,j].unsqueeze(1).unsqueeze(2)*((torch.cos(i*np.pi*X)*torch.cos(j*np.pi*Y)).unsqueeze(0))
        lossA += torch.mean(torch.norm(outA-solA[:,step+1,...],2,(-1,-2))/(torch.norm(solA[:,step+1,...],2,(-1,-2))))/num_steps
        loss0A += torch.mean(torch.norm(uA-targetA[:,step,...],2,(-1,-2))/(torch.norm(targetA[:,step,...],2,(-1,-2))))/num_steps
        ffA = (myallA[:,step+1,...]-myallA[:,step,...])/dt + DA*myallA[:,step+1,...]*eigen
        Non = net(myallA[:,step,...])
        loss1A += torch.mean(torch.norm(ffA - Non,2,(-1,-2))/(torch.norm(ffA,2,(-1,-2))))/num_steps
        u0A = uA
    print("{} & {:2.2e} & {:2.2e} & {:2.2e} \\\\".format(NN-9,loss0A,lossA,loss1A))
    # print("A: Train L2 loss: {:2.2e} True L2 loss: {:2.2e}, PINN loss: {:2.2e}".format(loss0A,lossA,loss1A))
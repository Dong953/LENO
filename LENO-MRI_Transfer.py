import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
class NonlinearNet(nn.Module):
    def __init__(self, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(p, M),
            nn.ReLU(),
            nn.Linear(M, p)
        )
        self.p = p
    def forward(self, u):
        return self.net(u)
device = 'cuda:4'
num_steps = 2
dt = T / num_steps
l0all = []
lall = []
l1all = []
p = 1024
M = 16
net = torch.load(net,f'ckpt/npde_ct_p_{p}_M_{M}.pth',map_location=device)
DA = torch.load(f'ckpt/npde_ct_DA_p_{p}_M_{M}.pth',map_location=device)
for NN in range(11,12):
    solA = np.load('./data/patient_all.npz')['res']
    betaA = np.load('./data/dat_coe.npz.npy')
    solA = torch.tensor(solA).to(device)
    betaA = torch.tensor(betaA).to(device)
    data_eigen = np.load('./data/eigen_brain_1024.npz')
    eigenf = data_eigen['eigenf']
    eigen = data_eigen['eigenv']
    mask_eval = data_eigen['mask_eval']
    ntrain = 1
    solA = solA[NN:NN+1,...]
    betaA = betaA[NN:NN+1,:,:p]
    targetA = betaA
    myallA = targetA[:ntrain,...]
    initialA = targetA[:ntrain,0,...]
    targetA = targetA[:ntrain,1:,...]
    eigenf = torch.tensor(eigenf).to(device)
    eigen = torch.tensor(eigen).to(device)
    eigen = eigen[:p]
    eigenf = eigenf[:p,:]
    Dt = nn.Parameter(torch.tensor(1.).double().to(device))
    
    for i, layer in enumerate(net.net):
        if isinstance(layer, nn.Linear) and i != len(net.net) - 1:  
            for param in layer.parameters():
                param.requires_grad = False
    optimizer = torch.optim.Adam(
        [
            {'params': [Dt]},  
            {'params': filter(lambda p: p.requires_grad, net.net.parameters())}  
        ],
        lr=lr,
        weight_decay=1e-8
    )
    epochs = 5000
    lr = 1e-3
    step_size = 1000
    gamma = 0.25
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_err = []
    for i in range(epochs):
        u0A = initialA
        loss = 0
        for step in range(num_steps):
            fcoe = net(u0A)
            uA = (u0A + Dt*dt*fcoe)/(1+Dt*DA*dt*eigen)
            loss += torch.mean(torch.norm(uA-targetA[:,step,...],2,(-1))/(torch.norm(targetA[:,step,...],2,(-1))+tol))/num_steps
            ffA = (myallA[:,step+1,...]-myallA[:,step,...])/(Dt*dt) + DA*myallA[:,step+1,...]*eigen
            Non = net(myallA[:,step,...])
            loss += torch.mean(torch.norm(ffA - Non,2,(-1))/(torch.norm(ffA,2,(-1))+tol))/num_steps
            u0A = uA
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_err.append(loss.item())
    with torch.no_grad():
        net.eval()
        num_steps = 2
        u0A = initialA
        lossA = 0
        loss0A = 0
        loss1A = 0
        for step in range(num_steps):
            fcoe = net(u0A)
            uA = (u0A + dt*fcoe)/(1+DA*dt*eigen)
            outA = torch.einsum('bi,ijk->bjk',uA,eigenf)
            lossA += torch.mean(torch.norm(outA-solA[:,step+1,...],2,(-1,-2))/(torch.norm(solA[:,step+1,...],2,(-1,-2))))/num_steps
            loss0A += torch.mean(torch.norm(uA-targetA[:,step,...],2,(-1))/(torch.norm(targetA[:,step,...],2,(-1))))/num_steps
            ffA = (myallA[:,step+1,...]-myallA[:,step,...])/dt + DA*myallA[:,step+1,...]*eigen
            Non = net(myallA[:,step,...])
            loss1A += torch.mean(torch.norm(ffA - Non,2,-1)/(torch.norm(ffA,2,-1)+tol))/num_steps
            u0A = uA
    print("{} & {:2.2f} & {:2.2f} & {:2.2f} \\\\".format(NN-10,100-100*loss0A,100-100*lossA,100-100*loss1A))
    l0all.append(loss0A.item())
    lall.append(lossA.item())
    l1all.append(loss1A.item())
print(" & {:2.2f} & {:2.2f} & {:2.2f} \\\\".format(100-100*np.mean(l0all),100-100*np.mean(lall),100-100*np.mean(l1all)))

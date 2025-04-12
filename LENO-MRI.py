import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:4'
num_steps = 2
dt = 1.
T = dt*num_steps
solA = np.load('./data/patient_all.npz')['res']
betaA = np.load('./data/dat_coe.npz.npy')
solA = torch.tensor(solA).to(device)
betaA = torch.tensor(betaA).to(device)
data_eigen = np.load('./data/eigen_brain_1024.npz')
eigenf = data_eigen['eigenf']
eigen = data_eigen['eigenv']
mask_eval = data_eigen['mask_eval']
mask_eval = torch.tensor(mask_eval).to(device)
solA = solA*mask_eval.unsqueeze(0).unsqueeze(0)
ntrain = 1
p = 1024
DA = nn.Parameter(torch.tensor(0.).double().to(device))
betaA = betaA[:ntrain,...,:p]
targetA = betaA
myallA = targetA[:ntrain,...]
initialA = targetA[:ntrain,0,...]
targetA = targetA[:ntrain,1:,...]
eigenf = torch.tensor(eigenf).to(device)
eigen = torch.tensor(eigen).to(device)
M = 16 
eigen = eigen[:p]
eigenf = eigenf[:p,:]
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
epochs = 5000
lr = 1e-3
lr_net = 1e-3
wd = 1e-8
wd_net = 1e-8
net = NonlinearNet(M,p).to(device)
net = net.double()
print(count_params(net))
optimizer = optim.Adam([
    {'params': net.parameters(), 'lr': lr_net, 'weight_decay': wd_net}, 
    {'params': DA, 'lr': lr, 'weight_decay': wd},
])
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_err = []
for i in range(epochs):
    u0A = initialA
    loss = 0
    for step in range(num_steps):
        fcoe = net(u0A)
        uA = (u0A + dt*alpha.unsqueeze(-1)*fcoe)/(1+alpha.unsqueeze(-1)*DA*dt*eigen)
        loss += torch.mean(torch.norm(uA-targetA[:,step,...],2,-1)/torch.norm(targetA[:,step,...],2,-1))/num_steps
        ffA = (myallA[:,step+1,...]-myallA[:,step,...])/alpha.unsqueeze(-1)/dt + DA*myallA[:,step+1,...]*eigen
        Non = net(myallA[:,step,...])
        loss += torch.mean(torch.norm(ffA - Non,2,-1))/num_steps
        u0A = uA
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    if (i+1) % 50 == 0:
       print(i,loss.item())
np.save('./result/loss_npde_ct.npy',np.array(train_err))
torch.save(net,f'ckpt/npde_ct_p_{p}_M_{M}.pth')
torch.save(DA,f'ckpt/npde_ct_DA_p_{p}_M_{M}.pth')

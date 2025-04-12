import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:1'
dt = .1
num_steps = 20
data = np.load('data/data_GS2d.npz')
solA = data['resA']
solS = data['resS']
DA = 2.5e-4
DS = 5e-4
num_points = solA.shape[-1]
x = np.linspace(0, 2*np.pi, num_points)
X,Y = np.meshgrid(x,x)
p = 48
M = 2000
ntrain = solA.shape[0]
betaA = np.zeros([solA.shape[0],solA.shape[1],p,p])
betaS = np.zeros([solS.shape[0],solS.shape[1],p,p])
nor = 2*np.ones([p])
nor[0] = 1.
eigen = np.zeros([p,p])
betaA = torch.tensor(betaA).to(device)
betaS = torch.tensor(betaS).to(device)
solA = torch.tensor(solA).to(device)
solS = torch.tensor(solS).to(device)
X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)
for i in range(p):
    for j in range(p):
        basis = torch.cos(i*X/2)*torch.cos(j*Y/2)
        Ac = solA*basis*nor[i]*nor[j]
        Sc = solS*basis*nor[i]*nor[j]
        Ac = .5*(Ac[...,:-1]+Ac[...,1:])
        Sc = .5*(Sc[...,:-1]+Sc[...,1:])
        Ac = .5*(Ac[...,:-1,:]+Ac[...,1:,:])
        Sc = .5*(Sc[...,:-1,:]+Sc[...,1:,:])
        betaA[...,i,j] = torch.mean(Ac,(2,3))
        betaS[...,i,j] = torch.mean(Sc,(2,3))
        eigen[i,j] = (i**2+j**2)/4
targetA = betaA
targetS = betaS
myallA = targetA[:ntrain,...]
initialA = targetA[:ntrain,0,...]
targetA = targetA[:ntrain,1:,...]
myallS = targetS[:ntrain,:,...]
initialS = targetS[:ntrain,0,...]
targetS = targetS[:ntrain,1:,...]
eigen = torch.tensor(eigen).to(device)
class NonlinearNet(nn.Module):
    def __init__(self, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(p*p*2, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, p*p*2)
        )
        self.p = p
    def forward(self, u):
        return self.net(u.view(*u.shape[:-3],-1)).view(*u.shape[:-3],2,self.p,self.p)

epochs = 5000
lr = 1e-3
net = NonlinearNet(M,p).to(device)
net = net.double()
optimizer = optim.Adam(net.parameters(), lr=lr)
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_err = []
for i in range(epochs):
    u0A = initialA
    u0S = initialS
    u0 = torch.cat((u0A.unsqueeze(1),u0S.unsqueeze(1)),dim=1)
    loss = 0
    for step in range(num_steps):
        fcoe = net(u0)
        uA = (u0A + dt*fcoe[:,0,...])/(1+DA*dt*eigen)
        uS = (u0S + dt*fcoe[:,1,...])/(1+DS*dt*eigen)
        loss += torch.mean(torch.norm(uA-targetA[:,step,...],2,(-1,-2))/(torch.norm(targetA[:,step,...],2,(-1,-2))))/num_steps
        loss += torch.mean(torch.norm(uS-targetS[:,step,...],2,(-1,-2))/(torch.norm(targetS[:,step,...],2,(-1,-2))))/num_steps
        ffA = (myallA[:,step+1,...]-myallA[:,step,...])/dt + DA*myallA[:,step+1,...]*eigen
        ffS = (myallS[:,step+1,...]-myallS[:,step,...])/dt + DS*myallS[:,step+1,...]*eigen
        myall = torch.cat((myallA[:,step,...].unsqueeze(1),myallS[:,step,...].unsqueeze(1)),dim=1)
        Non = net(myall)
        loss += torch.mean(torch.norm(ffA - Non[:,0,...],2,(-1,-2))/(torch.norm(ffA,2,(-1,-2))))/num_steps
        loss += torch.mean(torch.norm(ffS - Non[:,1,...],2,(-1,-2))/(torch.norm(ffS,2,(-1,-2))))/num_steps
        u0A = uA
        u0S = uS
        u0 = u0 = torch.cat((u0A.unsqueeze(1),u0S.unsqueeze(1)),dim=1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    if (i+1) % 50 == 0:
        print(f"Step {i+1}, Loss: {loss.item()}")
net.eval()
with torch.no_grad():
    u0A = initialA
    u0S = initialS
    u0 = torch.cat((u0A.unsqueeze(1),u0S.unsqueeze(1)),dim=1)
    lossA = 0
    lossS = 0
    loss1A = 0
    loss1S = 0
    for step in range(num_steps):
        fcoe = net(u0)
        uA = (u0A + dt*fcoe[:,0,...])/(1+DA*dt*eigen)
        uS = (u0S + dt*fcoe[:,1,...])/(1+DS*dt*eigen)
        outA = torch.zeros(solA.shape[0],solA.shape[-1],solA.shape[-1]).to(device)
        outS = torch.zeros(solA.shape[0],solA.shape[-1],solA.shape[-1]).to(device)
        for i in range(p):
            for j in range(p):
                outA += uA[:,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i)*X/2)*torch.cos((j)*Y/2)).unsqueeze(0))
                outS += uS[:,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i)*X/2)*torch.cos((j)*Y/2)).unsqueeze(0))
        lossA += torch.mean(torch.norm(outA-solA[:,step+1,...],2,(-1,-2))/(torch.norm(solA[:,step+1,...],2,(-1,-2))))/num_steps
        lossS += torch.mean(torch.norm(outS-solS[:,step+1,...],2,(-1,-2))/(torch.norm(solS[:,step+1,...],2,(-1,-2))))/num_steps
        ffA = (myallA[:,step+1,...]-myallA[:,step,...])/dt + DA*myallA[:,step+1,...]*eigen
        ffS = (myallS[:,step+1,...]-myallS[:,step,...])/dt + DS*myallS[:,step+1,...]*eigen
        myall = torch.cat((myallA[:,step,...].unsqueeze(1),myallS[:,step,...].unsqueeze(1)),dim=1)
        Non = net(myall)
        loss1A += torch.mean(torch.norm(ffA - Non[:,0,...],2,(-1,-2))/(torch.norm(ffA,2,(-1,-2))))/num_steps
        loss1S += torch.mean(torch.norm(ffS - Non[:,1,...],2,(-1,-2))/(torch.norm(ffS,2,(-1,-2))))/num_steps
        u0A = uA
        u0S = uS
        u0 = torch.cat((u0A.unsqueeze(1),u0S.unsqueeze(1)),dim=1)
    A = solA
    S = solS

    mu = .065
    rho = .04
    fA = S*A**2-(mu+rho)*A
    fS = -S*A**2+rho*(1.-S)
    beta = torch.cat((betaA.unsqueeze(2),betaS.unsqueeze(2)),dim=2)
    u = net(beta)
    outputA = torch.zeros_like(fA)
    outputS = torch.zeros_like(fS)
    for i in range(p):
           for j in range(p):
               outputA += u[:,:,0,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i)*X/2)*torch.cos((j)*Y/2)).unsqueeze(0))
               outputS += u[:,:,1,i,j].unsqueeze(-1).unsqueeze(-1)*((torch.cos((i)*X/2)*torch.cos((j)*Y/2)).unsqueeze(0))
    loss2A = torch.mean(torch.norm(outputA[:ntrain,:num_steps+1,...]-fA[:ntrain,:num_steps+1,...],2,(-2,-1)))/num_points**2*torch.pi*2
    loss3A = torch.mean(torch.norm(outputA[:ntrain,:num_steps+1,...]-fA[:ntrain,:num_steps+1,...],2,(-2,-1))/torch.norm(fA[:ntrain,:num_steps+1,...],2,(-2,-1)))
    loss2S = torch.mean(torch.norm(outputS[:ntrain,:num_steps+1,...]-fS[:ntrain,:num_steps+1,...],2,(-2,-1)))/num_points**2*torch.pi*2
    loss3S = torch.mean(torch.norm(outputS[:ntrain,:num_steps+1,...]-fS[:ntrain,:num_steps+1,...],2,(-2,-1))/torch.norm(fS[:ntrain,:num_steps+1,...],2,(-2,-1)))
    print("A: L2 loss: {:2.2e}, Residual loss: {:2.2e}, Nonlinear loss Absolute: {:2.2e},  Nonlinear loss Relative: {:2.2e}".format(lossA,loss1A,loss2A,loss3A))
    print("S: L2 loss: {:2.2e}, Residual loss: {:2.2e}, Nonlinear loss Absolute: {:2.2e},  Nonlinear loss Relative: {:2.2e}".format(lossS,loss1S,loss2S,loss3S))

import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utilities3 import *
device = 'cuda:2'
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

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cdouble))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cdouble)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x   
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
        
epochs = 5000
lr = 1e-3
net = FNO1d(16,p).to(device)
net = net.double()
print(count_params(net))
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
        uui = net(uini.unsqueeze(-1))
        uu = torch.zeros([ntrain,p]).double().to(device)
        for j in range(p):
            basis = uui.squeeze(-1)*torch.cos((j)*x/2)
            uu[:,j] = torch.mean(basis[:,1:]+basis[:,:-1],-1)*nor[j]/2
        u = (u0 + dt*uu)/(1+epsilon**2*dt*eigen)
        loss += torch.mean(torch.norm(u-target[:,step,:],2,-1)/torch.norm(target[:,step,:],2,-1))/num_steps
        Non = net(yall[:,step,:].unsqueeze(-1)).squeeze(-1)
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

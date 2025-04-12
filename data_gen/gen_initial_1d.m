% number of realizations to generate
N = 100;

% parameters for the Gaussian random field
gamma = 2.5;
tau = 7;
sigma = 7^(2);
% grid size
s = 1024;
steps = 1;


input = zeros(N, s+1);
x = linspace(0,1,s+1);
for j=1:N
    u0 = GRF1(s/2, 0, gamma, tau, sigma, "neumann");    
    u0eval = u0(x);
    input(j,:) = u0eval(1:end);
end

save('data_neu.mat','input')

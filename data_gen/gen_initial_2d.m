N = 100;
s = 257;
initialA = zeros(N,s,s);
initialS = zeros(N,s,s);
[X,Y] = meshgrid(0:(1/(s-1)):1);
alpha = 2.5;
tau = 7;
for i = 1:N
norm_a = GRF(alpha, tau, s);
initialA(i,:,:) = norm_a;
end
savemat('data_2d.mat','initialA')


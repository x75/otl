%generate data from multicariate normal
clear;
N=100;

mu = [0; 0];
st = [1; 2];
x(:,1) = normrnd(mu(1), st(1), N, 1);
x(:,2) = normrnd(mu(2), st(2), N, 1);
x(end,:) = [10, 10];

N = size(x,1);

scatter(x(:,1), x(:,2))

sigma = [st(1) 0; 0 st(2)];




m=N;
n=2;
% Find the maximum density value of the standard n-dimensional Gaussian
SqrtDet = sqrt(det(sigma));         % Find the sqrt of the determinant of the covariance matrix
C_n = (2*pi)^(n/2) .* SqrtDet;      % Eq. 13, the normalising coefficient
max_density(n) = 1/C_n;             % Find maximum pdf value, so that the x-axis can scale from [0 1]
invsigma = inv(sigma);
for i=1:N
    diff = (x(i,:) - mu');
    y(i) = (1/C_n)*exp(- 0.5*diff*invsigma*diff');
end

[c_m alpha_m] = EVT_GaussianEVD_FindParams(sigma, m);       % Find the EVD parameters for this Gaussian
% This uses G_n(y) to find c_m and
% g_n(y) and c_m to find alpha_m (see paper sec. 6.3)


Ge = EVT_GaussianEVD_Ge(y, c_m, alpha_m);                % Evaluate the EVD G_n^e(y) over the range of densities

%% plots
novelty_score = 1 - Ge;
novelty_score = novelty_score';

nsizes = novelty_score;
scatter3(x(:,1), x(:,2), novelty_score, 50, nsizes, 'f');
max(novelty_score)

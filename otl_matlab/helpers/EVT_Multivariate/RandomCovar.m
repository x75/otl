%% Create a random covariance matrix
% DC Logbook 22.142

function SIGMA = RandomCovar(n)

S = randn(n);
SIGMA = S'*S;
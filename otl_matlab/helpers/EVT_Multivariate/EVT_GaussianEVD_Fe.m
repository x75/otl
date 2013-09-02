%% Evaluate the pdf F_n^e(y) over Mahalanobis radii, r
% You might like to get the parameters c_m and alpha_m from EVT_GaussianEVT_FindParams.m
%
% DC Logbook 22.140
% Equations refer to Clifton et al. (2011), J. Sig. Proc. Sys. (65), pp. 371-389

function Fe = EVT_GaussianEVD_Fe(RS, SIGMA, c_m, alpha_m)

n = size(SIGMA,2);                  % Find the dimensionality
SqrtDet = sqrt(det(SIGMA));         % Find the sqrt of the determinant of the covariance matrix
C_n = (2*pi)^(n/2) .* SqrtDet;      % Eq. 13, the normalising coefficient

YS = (1/C_n).*exp(-(RS.^2)/2);      % Gaussian distribution in radius
Fe = exp(-(YS./c_m).^alpha_m);      % Eq. 32
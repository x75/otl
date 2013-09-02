%% Evaluate the pdf g_n(y) over densities, y
%
% DC Logbook 22.140
% Equations refer to Clifton et al. (2011), J. Sig. Proc. Sys. (65), pp. 371-389

function gy = EVT_GaussianPDF_gy(SIGMA, YS)

n = size(SIGMA,2);                  % Find the dimensionality
omega = 2*pi^(n/2)/gamma(n/2);      % Find the total solid angle subtended by the unit n-sphere, Omega_n
SqrtDet = sqrt(det(SIGMA));         % Find the sqrt of the determinant of the covariance matrix
C_n = (2*pi)^(n/2) .* SqrtDet;      % Eq. 13, the normalising coefficient

%% Evaluate the df at densities YS
gy = SqrtDet .* omega .* (-2*log(C_n .* YS)).^((n-2)/2);

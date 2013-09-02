%% Find the alpha (shape) and c (scale) parameters for the EVD pdf G_n^e(y) over densities, y
%
% DC Logbook 22.142
% Equations refer to Clifton et al. (2011), J. Sig. Proc. Sys. (65), pp. 371-389

function [c_m alpha_m] = EVT_GaussianEVD_FindParams(SIGMA, m)

%% Estimate the parameters of the Weibull
c_m = EVT_GaussianQuantile_Gy(SIGMA, 1/m);                  % Equation 28
alpha_m = m * c_m * EVT_GaussianPDF_gy(SIGMA, c_m);         % Equation 29
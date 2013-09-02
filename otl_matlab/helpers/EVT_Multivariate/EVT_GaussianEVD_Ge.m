%% Evaluate the pdf G_n^e(y) over densities, y
% You might like to get the parameters c_m and alpha_m from EVT_GaussianEVT_FindParams.m
%
% DC Logbook 22.140
% Equations refer to Clifton et al. (2011), J. Sig. Proc. Sys. (65), pp. 371-389

function Ge = EVT_GaussianEVD_Ge(YS, c_m, alpha_m)

Ge = 1 - exp(-(YS./c_m).^alpha_m);                      % Equation 30
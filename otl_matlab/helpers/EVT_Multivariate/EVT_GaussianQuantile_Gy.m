%% Find the p-quantile on the df over densities G_y
%
% DC Logbook 22.142
% Equations refer to Clifton et al. (2011), J. Sig. Proc. Sys. (65), pp. 371-389

function y = EVT_GaussianQuantile_Gy(SIGMA, p)

DOPLOT = false;

%% Find the maximum density ymax for this df
n = size(SIGMA,2);                  % Find the dimensionality
SqrtDet = sqrt(det(SIGMA));         % Find the sqrt of the determinant of the covariance matrix
C_n = (2*pi)^(n/2) .* SqrtDet;      % Eq. 13, the normalising coefficient
ymax = 1/C_n;                       % Find maximum pdf output value (i.e., density)

%% Find the p-quantile by finding the value of y that minimises |G_n(y) - p|
FirstGuess = ymax*p;        % Start at ymax * p, which is a good guess if G_n(y) is uniform...
y = fminsearch(@(y) abs(EVT_GaussianDF_Gy(SIGMA, y)-p), ymax*p);


if DOPLOT
    figure;
    YS = linspace(0, ymax, 10^6)';
    subplot(2, 1, 1);
    plot(YS, EVT_GaussianDF_Gy(SIGMA, YS));
    xlabel('y')
    ylabel('G_n(y)')
    subplot(2, 1, 2);
    plot(YS, abs(EVT_GaussianDF_Gy(SIGMA,YS)-p));
    xlabel('y');
    ylabel(sprintf('|G_n(y) - p|, p = %.3f |', p));
end
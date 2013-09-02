%% Plot figures that demonstrate multivariate EVT
% This shows pdfs g_n(y) over the densities, y, of an n-dimensional Gaussian distribution, f_n(x)
% and probability distributions G_n(y) over the same.
%
% Equations refer to Clifton et al. (2011), J. Sig. Proc. Sys. (65), pp. 371-389
% DC, Oct 2010

function Plot_GaussianDensities

%% Plot density g_n(y) of densities f_n(x) for standard multivariate Gaussian distributions (i.e., unit covariance)
figure;
for n = 1 : 6                           % Dimensionality of the Gaussian distribution, f_n(x)
    
    sigma = eye(n);                     % Covariance matrix of the standard n-dimensional Gaussian, f_n(x)
                                        % (Try with some non-isotropic Gaussians, but change max_density(n) accordingly)
    
    % Find the maximum density value of the standard n-dimensional Gaussian
    SqrtDet = sqrt(det(sigma));         % Find the sqrt of the determinant of the covariance matrix
    C_n = (2*pi)^(n/2) .* SqrtDet;      % Eq. 13, the normalising coefficient   
    max_density(n) = 1/C_n;             % Find maximum pdf value, so that the x-axis can scale from [0 1]                                      
    
    % Make the x-axis of our plot range from 0 up to this maximum density
    y{n} = linspace(0, max_density(n), 1e6)';

    % Find the density g_n(y) over density values f_n(x) over this range on the x-axis  (Eq. 20)
    gy = EVT_GaussianPDF_gy(sigma, y{n});
    
    % Now, print all graphs on the same axes
    g_sort = sort(gy);
    gmax = g_sort(round(0.999 * length(g_sort)));
    plot(y{n}./max_density(n), gy ./ gmax);          % Normalise the x-axis to have a maximum of unity
                                                     % and normalise each graph by its second-largest value
                                                     % (the largest typically being Inf!)
    hold on;
end
xlabel('y, normalised to 1 for comparison')
ylabel('g_n(y)')
set(gca, 'YLim', [0 1])


%% Plot distribution G_n(y) of densities f_n(x) for standard multivariate Gaussian distributions 
% (i.e., the integration of the above)

figure
for n = 1 : 6                                       % Dimensionality of the Gaussian distribution, f_n(x)

    sigma = eye(n);                                 % Covariance matrix of the n-dimensional Gaussian, f_n(x)

    Gy = EVT_GaussianDF_Gy(sigma, y{n});            % Find G_n(y)
    plot(y{n}./max_density(n), Gy);
    hold on;
end

xlabel('y, normalised to 1 for comparison')
ylabel('G_n(y)')

%% Plot the EVD, G_n^e(y)
figure
for n = 1 : 6
    
    sigma = eye(n);
    m = 50;
    
    [c_m alpha_m] = EVT_GaussianEVD_FindParams(sigma, m);       % Find the EVD parameters for this Gaussian
                                                                % This uses G_n(y) to find c_m and
                                                                % g_n(y) and c_m to find alpha_m (see paper sec. 6.3)
    Ge = EVT_GaussianEVD_Ge(y{n}, c_m, alpha_m);                % Evaluate the EVD G_n^e(y) over the range of densities
    semilogx(y{n}./max_density(n), Ge);                         % Plot the EVD G_n^e(y)
    hold on
end

xlabel('y, normalised to 1 for comparison')
ylabel('G_n^e(y)')
set(gca, 'YLim', [0 1]);

%% Run some numerical experiments, to determine if our EVDs G_n^e(y) over densities y were correct
% Use the value of m from before
figure
NUMSETS = 1e3;                          % Number of sets to generate (i.e., number of minima in our resultant plot)
YS = zeros(NUMSETS, 1);                 % For storing the minima, one per set
RS = zeros(NUMSETS, 1);                 % Corresponding (Mahalanobis) radii, one per set
for n = 1: 6
    for k = 1 : NUMSETS
        
        XS = gsamp(zeros(n, 1), eye(n), m);                     % Generate m data for this set (gsamp.m from Netlab)
        setYS = mvnpdf(XS, zeros(1,n), eye(n));                 % Find the densities for this set
        [minY minIdx] = min(setYS);                             % Find the most extreme (the minimum density)
        YS(k) = minY;                                           % Store this minimum for later
        RS(k) = mahalanobis(XS(minIdx,:), zeros(1,n),eye(n));   % Store this Mahalanobis radius for later
    end
    
    % Plot the results
    subplot(2, 1, 1)
    [c_m alpha_m] = EVT_GaussianEVD_FindParams(eye(n), m);      % Find the EVD parameters for this Gaussian
    Ge = EVT_GaussianEVD_Ge(y{n}, c_m, alpha_m);                % Evaluate the EVD G_n^e(y) over the range of densities
    semilogx(y{n}./max_density(n), Ge);                         % Plot the EVD G_n^e(y)
    hold on;
    [NS, YBIN] = hist(YS, 100);                                 % Find the histogram of our YS
    CDFNS = cumsum(NS./sum(NS));                                % Turn into an empirical df
    semilogx(YBIN./max_density(n), CDFNS, 'r.');
    
    subplot(2, 1, 2)
    r{n} = linspace(1, 6, 1e5)';                                % Mahalanobis radii, for plotting
    Fe = EVT_GaussianEVD_Fe(r{n}, eye(n), c_m, alpha_m);        % Evaluate the EVD F_n^e(r) over the range of radii
    plot(r{n}, Fe);
    hold on;
    [NS, RBIN] = hist(RS, 100);                                 % Find the histogram of our YS
    CDFNS = cumsum(NS./sum(NS));                                % Turn into an empirical df
    plot(RBIN, CDFNS, 'r.');    
end
subplot(2, 1, 1)
xlabel('y, normalised to 1 for comparison')
ylabel('G_n^e(y)')
set(gca, 'YLim', [0 1])
set(gca, 'XLim', [1e-5 1])
subplot(2, 1, 2)
xlabel('r, Mahalanobis radius')
ylabel('F_n^e(r)')
set(gca, 'YLim', [0 1])

%% Let's try the same for an n = 6 model, with an interesting covariance matrix
n = 6;
rng(7);                                                     % Set the random number generator's seed
SIGMA = RandomCovar(n)                                      % Create a "random" covariance matrix (using the rng seed)

for k = 1 : NUMSETS
    
    XS = gsamp(zeros(n, 1), SIGMA, m);                      % Generate m data for this set (gsamp.m from Netlab)
    setYS = mvnpdf(XS, zeros(1,n), SIGMA);                  % Find the densities for this set
    [minY minIdx] = min(setYS);                             % Find the most extreme (the minimum density)
    YS(k) = minY;                                           % Store this minimum for later
    RS(k) = mahalanobis(XS(minIdx,:), zeros(1,n), SIGMA);   % Store this Mahalanobis radius for later
end

% Plot the results
figure
subplot(2, 1, 1)
[c_m alpha_m] = EVT_GaussianEVD_FindParams(SIGMA, m);       % Find the EVD parameters for this Gaussian
Ge = EVT_GaussianEVD_Ge(y{n}, c_m, alpha_m);                % Evaluate the EVD G_n^e(y) over the range of densities
semilogx(y{n}./max_density(n), Ge);                         % Plot the EVD G_n^e(y)
hold on;
[NS, YBIN] = hist(YS, 100);                                 % Find the histogram of our YS
CDFNS = cumsum(NS./sum(NS));                                % Turn into an empirical df
semilogx(YBIN./max_density(n), CDFNS, 'r.');
set(gca, 'YLim', [0 1])
set(gca, 'XLim', [1e-6 1e-3])
xlabel('y, normalised to 1 for comparison')
ylabel('G_n^e(y)')

subplot(2, 1, 2)
r{n} = linspace(1, 6, 1e5)';                                % Mahalanobis radii, for plotting
Fe = EVT_GaussianEVD_Fe(r{n}, SIGMA, c_m, alpha_m);         % Evaluate the EVD F_n^e(r) over the range of radii
plot(r{n}, Fe);
hold on;
[NS, RBIN] = hist(RS, 100);                                 % Find the histogram of our YS
CDFNS = cumsum(NS./sum(NS));                                % Turn into an empirical df
plot(RBIN, CDFNS, 'r.');
set(gca, 'YLim', [0 1])
set(gca, 'XLim', [3 6])
xlabel('r, Mahalanobis radius')
ylabel('F_n^e(r)')

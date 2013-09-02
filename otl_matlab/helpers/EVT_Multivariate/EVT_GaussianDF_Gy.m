%% Evaluate the df G_n(y) over densities, y
%
% DC Logbook 22.140
% Equations refer to Clifton et al. (2011), J. Sig. Proc. Sys. (65), pp. 371-389

function Gy = EVT_GaussianDF_Gy(SIGMA, YS)

%% Initialisation
if size(YS, 1) == 1
    YS = YS';               % Transpose into a column vector, if necessary
end
Gy = nan(length(YS), 1);

%% We need to check that the input isn't out of range; if they are, we can treat them immediately

n = size(SIGMA,2);                  % Find the dimensionality
SqrtDet = sqrt(det(SIGMA));         % Find the sqrt of the determinant of the covariance matrix
C_n = (2*pi)^(n/2) .* SqrtDet;      % Eq. 13
ymax = 1/C_n;                       % Find maximum pdf value, so that the x-axis can scale from [0 1]

IS = find((YS > 0) & (YS <= ymax)); % Only find the values for the non-limiting densities

if length(IS) > 0
    if n == 1                           % Use Eq.21 for the univariate case
        
        Gy(IS) =  erfc(sqrt(-log(C_n .* YS(IS))));
        
    else
        if (mod(n,2) == 0)              % if n is even, use Eqs. 22 and 24
            
            p = n/2;                    % Find the upper limit of p (n = 2p)
            ks = 0 : (p-1);             % Indices ks for the summations in Eq. 22
            
            omega = 2*pi^(n/2)/gamma(n/2);      % Find the total solid angle subtended by the unit n-sphere, Omega_n
            C_2p = SqrtDet*(2*pi)^(n/2);        % Eq. 13 again
            
            % Now use Eqs. 22 and 24
            A_CommonTerm = omega .* SqrtDet;    % Calculate the common term from Eq. 24
            A_SumTerms = (2.^ks .* factorial(p-1)) ./factorial(p-ks-1);     % Calculate the series A in Eq. 24
            A_SumTerms = A_SumTerms .* A_CommonTerm;                        % Multiply each A by the common term
            
            G = zeros(length(IS), 1);   % Eq. 22: sum G over the various terms in the series A
            for i_ks = 1 : length(ks)   % Range over the indices in ks
                % (i.e., i_ks = 1 when ks = 0, and i_ks = max when ks = p-1
                G = G + A_SumTerms(i_ks) .* (-2 .* log(C_2p .* YS(IS))).^(p-ks(i_ks)-1);
            end
            Gy(IS) = G .* YS(IS);               % Complete Eq.22 by multiplying by y
            
        else                            % n is odd, so use Eqs. 23 and 25
            
            p = floor(n/2);             % Find the upper limit of p (n = 2p+1)
            ks = 0 : (p-1);             % Indices ks for the summations in Eq. 23
            
            omega = 2*pi^(n/2)/gamma(n/2);      % Find the total solid angle subtended by the unit n-sphere, Omega_n
            C_2p1 = SqrtDet*(2*pi)^(n/2);       % Eq. 13 again
            
            % Now use Eqs. 23 and 25
            A_CommonTerm = omega .* SqrtDet;    % Calculate the common term from Eq. 24
            A_SumTerms = (factorial(2*p-1) .* factorial(p-ks)) ./ ...
                (2.^(ks-1) .* factorial(p-1) .* factorial(2*p - 2*ks));     % Calculate the series A in Eq. 25
            A_SumTerms = A_SumTerms .* A_CommonTerm;                        % Multiply each A by the common term
            
            G = zeros(length(IS), 1);   % Eq. 22: sum G over the various terms in the series A
            for i_ks = 1 : length(ks)   % Range over the indices in ks
                % (i.e., i_ks = 1 when ks = 0, and i_ks = max when ks = p-1
                G = G + A_SumTerms(i_ks) .* (-2 .* log(C_2p1 .* YS(IS))).^(p-ks(i_ks)-1/2);
            end
            Gy(IS) = G .* YS(IS) + erfc(sqrt(-log(C_2p1 .* YS(IS))));                % Complete Eq.22 by multiplying by y
        end
    end
end
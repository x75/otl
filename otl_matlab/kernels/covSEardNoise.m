function K = covSEardNoise(hyp, x, z, i)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sqrt(sf2)) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.
tol = 1e-9;  % threshold on the norm when two vectors are considered to be equal
if nargin<2, K = '(D+2)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[~,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance
s2 = exp(2*hyp(D+2));               %noise variance
n = size(x,1);


% precompute squared distances
if dg                                                               % vector kxx
    K = zeros(size(x,1),1) + ones(n,1);
    K2 = ones(n,1);
else
    if xeqz                                                 % symmetric matrix Kxx
        K = sq_dist(diag(1./ell)*x');
        K2 = eye(n);
    else                                                   % cross covariances Kxz
        K = sq_dist(diag(1./ell)*x',diag(1./ell)*z');
        K2 = double(sq_dist(x',z')<tol*tol);
    end
end

K1 = sf2*exp(-K/2);

if nargin <=3
    K = K1 + s2*K2;
end

if nargin>3                                                        % derivatives
    if i<=D                                              % length scale parameters
        if dg
            K = zeros(size(K));
        else
            if xeqz
                K = K1.*sq_dist(x(:,i)'/ell(i));
            else
                K = K1.*sq_dist(x(:,i)'/ell(i),z(:,i)'/ell(i));
            end
        end
        
    elseif i==D+1                                            % magnitude parameter
        K = 2*K1;
    elseif i==D+2
        K = 2*s2*K2;
    else
        error('Unknown hyperparameter')
    end
end


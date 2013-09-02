function K = covRecSEisoNoise(hyp, x, z, hd)

% Recursive Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) =  exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         0.5*log(sf2)
%         tau
%         rho^2
%         log(noise)   ]
%
% Copyright (c) by Harold Soh, 2012.
%
% See also COVFUNCTIONS.M.

if nargin<2, K =sprintf('(5)'); return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

ell = exp(hyp(1));                               % characteristic length scale

sf2 = exp(2*hyp(2));
rho = hyp(3);% 1/(1+exp(-ori_rho));
s2 = exp(2*hyp(4)); %noise parameter
tau = round(hyp(5));

%noise params
tol = 1e-9;  % threshold on the norm when two vectors are considered to be equal
m = size(x,2);
n = size(x,1);
D = m/tau;
ell = ell*ones(D,1);

%% go through all the items to compute the kernel matrices
% compute recursive covariance
if nargin <=3
    
    %recursive terms
    if dg
        CRK_tau = ones(n,1);
    else
        if xeqz
            CRK_tau = getRecKernMat(ell, tau, rho, 0, x');
        else
            CRK_tau = getRecKernMat(ell, tau, rho, 0, x', z');
        end
    end
    K1 = sf2*CRK_tau;
    
    %% noise terms
    if dg                                                               % vector kxx
        K2 = ones(n,1);
    else
        if xeqz                                                 % symmetric matrix Kxx
            K2 = eye(n);
        else                                                   % cross covariances Kxz
            K2 = double(sq_dist(x',z')<tol*tol);
        end
    end
    
    K2 = s2*K2;
    
    %% Sum of kernels
    K = K1 + K2;
    
    
end

%% derivative computations

if nargin>3                                                        % derivatives
    if hd==1                                              % length scale parameters
        
        mod_hd = 1; %because i don't want to have to recode the same thing
        if dg
            K = K*0;
        else
            if xeqz
                K = getRecKernMat(ell, tau, rho, mod_hd, x');
            else
                K = getRecKernMat(ell, tau, rho, mod_hd, x', z');
            end
            
            K = sf2*K;
            
        end
    elseif hd==2                                         % magnitude parameter
        
        
        if dg
            CRK_tau = ones(D,1);
        else
            if xeqz
                CRK_tau = getRecKernMat(ell, tau, rho, 0, x');
            else
                CRK_tau = getRecKernMat(ell, tau, rho, 0, x', z');
            end
        end
                
        K = 2*sf2*CRK_tau;
        
    elseif hd==3 %rho parameter
        %K = K*0;
        %if tau == 1
        %    K = K*0;
        %else
        mod_hd = D+2;
        if xeqz
            K = getRecKernMat(ell, tau, rho, mod_hd, x');
        else
            K = getRecKernMat(ell, tau, rho, mod_hd, x', z');
        end
        K = sf2*K;
        %end
    elseif hd==4 %noise parameter
        
        if dg                                                               % vector kxx
            K2 = ones(n,1);
        else
            if xeqz                                                 % symmetric matrix Kxx
                K2 = eye(n);
            else                                                   % cross covariances Kxz
                K2 = double(sq_dist(x',z')<tol*tol);
            end
        end
        
        K = 2*s2*K2;
    elseif hd == 5 %tau parameter (don't optimise this
        if dg                                                               % vector kxx
            K = zeros(n,1);
        else
            if xeqz                                                 % symmetric matrix Kxx
                K = zeros(n);
            else                                                   % cross covariances Kxz
                K = zeros(size(x',1), size(z',1));
            end
        end
        warning('You should not be trying to optimise this parameter');
    else
        hd
        error('Unknown hyperparameter')
    end
end
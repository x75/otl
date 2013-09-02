function K = covTempSEardNoise(hyp, x, z, hd)

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
global rD_recKern_7389; %we need this global variable because of how these functions work.
if nargin<2, K =sprintf('(%d+4)', rD_recKern_7389); return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

tD = length(hyp);
tau = round(hyp(tD));
D = round(tD - 3 - tau);

ell = exp(hyp(1:D));                              % characteristic length scale
%invellD = (1./ell);
rho = exp(2*hyp(D+1:D+tau));
%invrhosD = (1./rhos);
sf2 = exp(2*hyp(D+tau+1));
s2 = exp(2*hyp(D+tau+2)); %noise parameter

%noise params
tol = 1e-9;  % threshold on the norm when two vectors are considered to be equal
n = size(x,1);


%% go through all the items to compute the kernel matrices
% compute recursive covariance
if nargin <=3
    
    %recursive terms
    if dg
        CRK_tau = ones(D,1);
    else
        if xeqz
            
            CRK_tau = getTempSEMat(ell, tau, rho, 0, x');
        else
            CRK_tau = getTempSEMat(ell, tau, rho, 0, x', z');
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
    if hd<=D                                              % length scale parameters
        if dg
            K = K*0;
        else
            if xeqz
                K = getTempSEMat(ell, tau, rho, hd, x');
            else
                K = getTempSEMat(ell, tau, rho, hd, x', z');
            end
            
            K = sf2*K;
            
        end
    elseif hd==D+tau+1                                            % magnitude parameter
        if dg
            CRK_tau = ones(D,1);
        else
            if xeqz
                CRK_tau = getTempSEMat(ell, tau, rho, 0, x');
            else
                CRK_tau = getTempSEMat(ell, tau, rho, 0, x', z');
            end
        end
        
        %K = 2*sf2*CRK_tau;
        K=2*sf2*CRK_tau;
    elseif any(hd==[D+1:D+tau]) %rho parameter
        %K = K*0;
        %if tau == 1
        %    K = K*0;
        %else
        if xeqz
            K = getTempSEMat(ell, tau, rho, hd, x');
        else
            K = getTempSEMat(ell, tau, rho, hd, x', z');
        end
        K = sf2*K;
        %end
    elseif hd==D+tau+2 %noise parameter
        
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
    elseif hd==D+tau+3 %tau parameter
        
        if dg                                                               % vector kxx
            K2 = ones(n,1);
        else
            if xeqz                                                 % symmetric matrix Kxx
                K2 = eye(n);
            else                                                   % cross covariances Kxz
                K2 = double(sq_dist(x',z')<tol*tol);
            end
        end
        
        K = K2*0;    
    
    else
        error('Unknown hyperparameter')
    end
end
function [ K ] = getTempSEMat( ells, tau, rhos, hd, x, z )
%GETTEMPSEMAT Summary of this function goes here
%   Detailed explanation goes here
noz = false;
if nargin == 5
    noz = true;
end

D = length(ells);
tau = length(rhos);
%
%     [xd, xr] = size(x);
%     [zd, zr] = size(z);
%
%     if xd ~= zd || xd ~= D
%         error('Wrong dimensions');
%     end

%create new ellrhos
ellrhos = []; %zeros(D*tau);
for i=1:tau
    ellrhos = [ellrhos; ells.*rhos(i)];
end
ellrhosD = diag(1./ellrhos);
%     length(ellrhos)
%     length(x)
%we just want to compute the kernel

if noz
    K = sq_dist(ellrhosD*x);
else
    K = sq_dist(ellrhosD*x,ellrhosD*z);
end

K = exp(-K/2);


if hd == 0
    return;
end

%we want to compute derivates
if any(hd==[D+1:D+tau])
    %rho parameters
    
    pid = hd - D -1; %which parameter?
    hds = [(pid*D)+1:(pid*D)+D];
    
    X = x(hds, :);
    
    X = X ./ repmat(ellrhos(hds), 1, size(X,2));
    
    if noz
        K = K.*sq_dist(X);
    else
        Z = z(hds, :);
        Z = Z ./ repmat(ellrhos(hds), 1, size(Z,2));
        %Z = z(hds, :) ./ ellrhos(hds);
        
        K = K.*sq_dist(X,Z);
    end
elseif hd<=D
    %lengthscale parameters
    
    
    hds = [hd:D:D*tau];
    
    X = x(hds, :);
    X = X ./ repmat(ellrhos(hds), 1, size(X,2));
    
    if noz
        K = K.*sq_dist(X);
    else
        Z = z(hds, :);
        Z = Z ./ repmat(ellrhos(hds), 1,size(Z,2));
        K = K.*sq_dist(X,Z);
    end
end


end


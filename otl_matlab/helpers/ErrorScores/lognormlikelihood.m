function [ sumloglik, loglik ] = lognormlikelihood( desired, pred_mean, pred_var )
%LOGNORMLIKELIHOOD Summary of this function goes here
%   Detailed explanation goes here
    
    loglik = - 0.5*log(pred_var) - (((desired - pred_mean).^2) ./ (2*pred_var)) - ...
        0.5*log(2*pi);
    
    sumloglik = sum(loglik);
end


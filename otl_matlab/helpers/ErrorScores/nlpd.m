function [ err ] = nlpd(  desired, pred_mean, pred_var )
%NLPD Negative log predictive density score
%   Assumes gaussian likelihood density
    err = 0.5*log(pred_var) + (((desired - pred_mean).^2) ./ (2*pred_var)) + ...
        0.5*log(2*pi);
    
    %we simply get the mean across all dimensions
    err = nanmean(nanmean(err));
   
end


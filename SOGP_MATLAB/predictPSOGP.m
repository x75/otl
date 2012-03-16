function [ pred_mean, pred_var ] = predictPSOGP( x, psogp )
%PREDICT_PSOGP Summary of this function goes here
%   Detailed explanation goes here
    kstar = getKStar(x, psogp);
    
    if not(isfield(psogp, 'phi')) 
        pred_mean = 0;
        pred_var = kstar + psogp.noise;
        return
    end
    
    k = getKStarVector(x, psogp);
    pred_mean = k'*psogp.alpha;
    pred_var = psogp.noise + kstar + k'*(psogp.C*k);

end


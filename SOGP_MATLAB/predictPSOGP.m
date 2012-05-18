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
    
    if psogp.type == 'r'
        pred_mean = k'*psogp.alpha;
        pred_var = psogp.noise + kstar + k'*(psogp.C*k);
    elseif psogp.type == 'c'
        pred_mean = k'*psogp.alpha
        pred_var = psogp.noise + kstar + k'*(psogp.C*k)
        psogp.C
        sx = sqrt((pred_var))
        z = pred_mean/sx
        
        Erfz = Erf(z);
        pred_mean = round(Erfz)*2 - 1;
        pred_var = Erfz;
    end
    

end

 function erfz = Erf(z)
    erfz = normcdf(z,0,1);
 end


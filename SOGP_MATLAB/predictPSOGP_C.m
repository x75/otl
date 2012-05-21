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
        
        for cl=1:psogp.num_classes
        
            pred_mean = k'*psogp.alpha{cl};
            pred_var = psogp.noise + kstar + k'*(psogp.C{cl}*k);

            sx = sqrt((pred_var));
            z(cl) = pred_mean/sx;
        end
        
        Erfz = Erf(z);
        
        for i=1:size(Erfz,2)
            if Erfz(:,i) > 0.5
                pred_mean(:,i) = 1;
            else
                pred_mean(:,i) = -1;
            end            
        end
        pred_mean;

        pred_var = Erfz;
    end
    

end

 function erfz = Erf(z)
    erfz = normcdf(z,0,1);
 end


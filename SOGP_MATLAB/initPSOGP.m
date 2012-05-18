function [ psogp ] = initPSOGP( params, kernFunc, kernParams, problemType )
%TRAIN_PSOGP Summary of this function goes here
%   Detailed explanation goes here
    psogp = struct();
    
    psogp.capacity = params(1);
    psogp.noise = params(2);
    psogp.epsilon = params(3);
    psogp.covf = kernFunc;
    psogp.covf_params = kernParams;
    
    psogp.alpha = 0;
    psogp.C = 0;
    
    %problemType is 'regression' or 'classification'
    psogp.type = problemType;
    
end


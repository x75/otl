function [ psogp ] = initPSOGP_C( params, kernFunc, kernParams, ... 
    num_classes, deletionCriteria )
%TRAIN_PSOGP Summary of this function goes here
%   Detailed explanation goes here
    psogp = struct();
    
    psogp.capacity = params(1);
    psogp.noise = params(2);
    psogp.epsilon = params(3);
    psogp.covf = kernFunc;
    psogp.covf_params = kernParams;
    psogp.num_classes = num_classes;
    
    psogp.alpha = {};
    psogp.C = {};
        
    %deletion criteria is 'n' (norm) or 'm' (minimax)
    psogp.deletion_criteria = deletionCriteria;
    
end


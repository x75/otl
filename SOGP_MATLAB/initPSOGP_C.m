function [ psogp ] = initPSOGP_C( params, kernFunc, kernParams, ... 
    problemType, deletionCriteria, num_classes)
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
    
    %problemType is 'r' (regression) or 'c' (classification)
    psogp.type = problemType;
    
    %deletion criteria is 'n' (norm) or 'm' (minimax)
    psogp.deletion_criteria = deletionCriteria;
    
end


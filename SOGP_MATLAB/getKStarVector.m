function [ kstarv ] = getKStarVector( x, PSOGP )
%GETKSTAR Summary of this function goes here
%   Detailed explanation goes here
    lenK = size(PSOGP.phi,2);
    kstarv = zeros(lenK,1);
    for i = 1:lenK    
        kstarv(i) = PSOGP.covf(x, PSOGP.phi{i}, PSOGP.covf_params);
    end   
end


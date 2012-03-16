function [ kstar ] = getKStar( x, PSOGP )
%GETKSTAR Summary of this function goes here
%   Detailed explanation goes here
    kstar = PSOGP.covf(x, x, PSOGP.covf_params); 
end


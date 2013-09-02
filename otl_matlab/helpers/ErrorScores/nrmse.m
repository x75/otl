function [ err ] = nrmse( desired, predicted )
%NMSE returns the root-mean squared error 
%   rows are considered samples,
    N = size(desired,1);
    
    temp = desired - predicted;
    
    temp = mean(temp.*temp);
        
    err = sqrt(temp/var(desired));
end

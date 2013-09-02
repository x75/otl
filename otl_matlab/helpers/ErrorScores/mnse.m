function [ err ] = mnse( desired, predicted )
%RMSE returns the root-mean squared error 
%   rows are considered samples,
    N = size(desired,1);
    
    d2 = sum(desired.*desired);
    temp = desired - predicted;
    temp = sum(temp.*temp);
    
    err = (temp/d2);
end


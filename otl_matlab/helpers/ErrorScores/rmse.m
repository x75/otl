function [ err ] = rmse( X, Y )
%RMSE returns the root-mean squared error 
%   rows are considered samples, and columns are considered dimensions
    N = size(X,1);
    D = size(X,2);
    temp = X - Y;
    temp = temp.*temp;
    temp = sum(temp, 2)/D;
    err = sqrt(sum(temp)/(N-1));
end


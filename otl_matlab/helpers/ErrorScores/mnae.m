function [ err , all_errs] = mnae( desired, predicted )
%RMSE returns the root-mean squared error 
%   rows are considered samples,
    N = size(desired,1);
    D = size(desired,2);
    s2 = var(desired);
    
    temp = (desired - predicted).^2;
    temp = temp ./ repmat(s2, N, 1);
    temp = sum(temp, 2)/D;
    
    temp = sqrt(temp);
    all_errs = temp;
    err = sum(temp)/N;
end


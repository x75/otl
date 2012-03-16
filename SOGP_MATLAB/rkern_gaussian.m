function [ kval ] = rkern_gaussian( x, y, params )
%RKERN_GAUSSIAN Summary of this function goes here
%   Detailed explanation goes here
    sigmai = params(1);
    sigma = params(2);
    leak_rate = params(3);
    ret_rate = 1-leak_rate;
    T = size(x,1);
    kval = 0;
    for i=1:T
        kval = exp(- norm( ret_rate*(x(i,:) - y(i,:)) )^2/(2*sigmai^2))* ...
            exp((kval -1.0)/(sigma^2));
    end
end


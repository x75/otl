function [ kval ] = kern_gaussian( x, y, params )
%RKERN_GAUSSIAN Summary of this function goes here
%   Detailed explanation goes here
    sigmai = params(1);
    kval = exp(- norm((x - y) )^2/(2*sigmai^2));

end


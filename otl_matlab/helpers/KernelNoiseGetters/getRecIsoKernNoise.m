function [ noise ] = getRecIsoKernNoise( hyparams )
%GETRECKERNNOISE Summary of this function goes here
%   Detailed explanation goes here
    noise = exp(2*hyparams(end-1));
end


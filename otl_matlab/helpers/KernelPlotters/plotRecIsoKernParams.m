function [kernparams] = plotRecIsoKernParams( figh, params )
%PLOTTEMPKERNPARAMS plots the parameters of the temporal kernel
if nargin == 1
    figure();
    params = figh;
else
    figure(figh);
end

kern_params = params;
x_dim = 1;
ell_t = exp(kern_params(1:x_dim));
sf2_t = exp(2*kern_params(x_dim+1));
rho_t = kern_params(x_dim+2);
rho_t = rho_t;
lognoise_t = exp(2*kern_params(end-1));

kernparams = [ell_t; sf2_t; rho_t; lognoise_t];
bar([ell_t; sf2_t; rho_t; lognoise_t]);
set(gca, 'Yscale', 'log');
ylim([10^-7, 10^3]);

labels = cell(size(kern_params));
for i=1:x_dim
    labels{i} = sprintf('l_%d', i);
end
labels{x_dim+1} = 'sf2';
labels{x_dim+2} = 'rho2';
labels{x_dim+3} = 'noise';

xlabel('Hyperparameters');
set(gca,'XTickLabel',labels);


drawnow();

end


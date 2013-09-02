function plotTempKernParams( figh, params )
%PLOTTEMPKERNPARAMS plots the parameters of the temporal kernel

    %figure(figh);
    hyp = params;
    tD = length(hyp);
    tau = round(hyp(tD));
    D = round(tD - 3 - tau);

    ell = exp(hyp(1:D));                              % characteristic length scale
    %invellD = (1./ell);
    rho = exp(hyp(D+1:D+tau));
    %invrhosD = (1./rhos);
    sf2 = exp(2*hyp(D+tau+1));
    s2 = exp(2*hyp(D+tau+2)); %noise parameter

    figure(figh);
    bar([ell; rho; sf2; s2]);
    set(gca, 'Yscale', 'log');
    ylim([10^-7, 10^5]);

    labels = cell(size(hyp)-1);
    for i=1:D
        labels{i} = sprintf('l_%d', i);
    end
    for i=D+1:D+tau
        labels{i} =  sprintf('p_%d', i);
    end
    labels{D+tau+1} = 'sf2';
    labels{D+tau+2} = 'noise';

    %xlabel('Hyperparameters');
    set(gca,'XTickLabel',labels);
    %title(figh,sprintf('Model: %d, Optimization Iteration: %d', stork.model_id));
    drawnow();

end


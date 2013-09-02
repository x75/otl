%NoisySineCos_MemOGP2_Opt d.m
%Author: Harold Soh
%This script gives an example of how to use the Memory Online Gaussian
%Process (MemOGP2). Notice how (unlike regular batch learning), we train and
%predict iteratively. This also shows you how to set the optimisation
%parameters to perform hyperparmeter optimisation

%% Experimental parameters
clear();

s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);
do_update_plot = false;
do_final_plot = true;
debug_mode = false;

%problem parameters
n_extra_useless_dimensions = 2; %to test the ARD
noise = sqrt(0.01);
N = 2000;
dt = 0.05; %time difference

%optimisation parameters
optimize_params = true;
opt_interval = 10;


%% Set up our data
D = sin(0:dt:N*dt)';
Y_true = D(2:end);
D = D + normrnd(0,noise,size(D));
D = [D 5*rand(size(D,1), n_extra_useless_dimensions)];

%% Create the MemOGP (In this case the OIESGP)
x_dim = n_extra_useless_dimensions + 1;
y_dim = 1;

%first specify our parametres
capacity = 200;       %30 basis vectors
gp_noise = noise;      %prior noise of 0.1
epsilon = 1e-5;     %tolerance parameter

%create our memory object
memory_length = 10;
dl = FDelayLine(x_dim, memory_length);

%we need to be careful while specifying the dimensions of the
%hyperparameters. This should be dimension of the MEMORY and not the input
kern_func = {'covRecSEardNoise'}

ell = zeros(x_dim,1) + log(10.0);
max_ell = zeros(x_dim,1) + log(1000000.0);
min_ell = zeros(x_dim,1) + log(0.001);

sf2 = 0.5*log(0.5);             %signal variance parameter (log)
min_sf2 = 0.5*log(0.001);
max_sf2 = 0.5*log(10.0);

tau = memory_length;

rho = 0.9;
min_rho = 0.5;
max_rho = 1.1;

lognoise = 0.5*log(noise);
min_lognoise = 0.5*log(0.001);
max_lognoise = 0.5*log(0.5);


kern_params = [ell; sf2; rho; lognoise];    %kernel parameters
lower_bounds = [min_ell; min_sf2; min_rho; min_lognoise];
upper_bounds = [max_ell; max_sf2; max_rho; max_lognoise];
%kern_params = [ell; sf2; lognoise];

pm_hyp_to_optimise = 1:length(kern_params);
pm_num_opt_per_itr = length(kern_params);
pm_update_thres = 0.01;
pm_stop_opt_thres = 5000;

n_kern_params = length(kern_params);

rfilt = 0.8; %filter for recency scores
sw = 1.0;      %weights between recency and reconstruction
regularizer = 1.0; %regularizer for p and q

lambda0 = 0.01;
multiplier = 1.0;
%gradient_bound = 'auto';
gradient_bound = 0.1;
use_evt = false;
evt_threshold = 0.999;

% error or likelihood based optimisation
%pm_lik_mode = 'e'; %gives an final nmse of ~ 0.0281 
pm_lik_mode = 'p'; %gives a final nmse of ~ 0.0695
noisefunc = @getRecKernNoise;

gp_params = [capacity, gp_noise, epsilon, rfilt, sw, regularizer];
plotfunc = @plotRecKernParams;
opt_params = {opt_interval, lambda0, multiplier, gradient_bound, ...
    lower_bounds, upper_bounds, plotfunc, do_update_plot...
    pm_hyp_to_optimise, pm_num_opt_per_itr, ...
    pm_update_thres, pm_stop_opt_thres, pm_lik_mode};

%initialise the MemOGP object
inf_mode = 'r'; %'r' for regression, 'c' for classification
pMemOGP = FMemOGP2f(inf_mode,gp_params, dl, ...
    kern_func, kern_params, noisefunc,...
    opt_params, 'n', debug_mode);

pMemOGP.setHypOpt(optimize_params);


%% Run our little experiment
%initialise our results storage
y_pred = [];
y_var = [];
hyp_gradients = [];
log_marg_lik = zeros(N,y_dim);

%for each data item, add it to the MemOGP
if do_update_plot
    figure();
end

trainEnd = ceil(0.8*N);

%% training for the first 80%
for i=1:trainEnd %size(X,1)
    tic;
    
    X(i,:) = D(i,:);
    Y(i,:) = D(i+1,1);
    nbvs = pMemOGP.getNumBasisVectors();
    fprintf('Iteration %d, num BVs: %d\n', i, nbvs);
    pMemOGP.updateMemory(X(i,:));
    [y_pred(i,:), y_var(i,:), log_marg_lik(i,:), MemOGP_rebuilt] = pMemOGP.trainInterval(Y(i,:));
    
    all_nbvs(i) = nbvs;
    log_marg_lik(i,:) = log(normpdf(Y_true(i,:), y_pred(i,:), sqrt(y_var(i,:))));
    
    if i == ceil(trainEnd/2)
        pMemOGP.setHypOpt(false);
    end
    
    kern_params = pMemOGP.getKernParams();

    time_taken(i) = toc();
end

%% testing for the final 20%
for i=trainEnd+1:N %size(X,1)
    tic;
    fprintf('Iteration %d\n', i);
    
    X(i,:) = D(i,:);
    Y(i,:) = D(i+1,1);
    
    pMemOGP.updateMemory(X(i,:));
    [y_pred(i,:), y_var(i,:)] = pMemOGP.predict();
    time_taken(i) = toc();
    
end

%% Plot each dimension in its own plot
if do_final_plot
    figure()
    y_dim = size(Y,2);
    for i=1:y_dim
        subplot(y_dim+2,1,i);
        errorbar(1:N, y_pred(1:N,i), y_var(1:N), 'b-'); hold on;
        scatter(1:size(y_pred,1), Y(1:size(y_pred,1),i), 'g+');
        ylim([-1.5, 1.5]);
        title(sprintf('Y(%d): All Data', i));
        hold off;
    end
    
    %% plot the absolute error
    subplot(y_dim+2,1,y_dim+1);
    
    for i=1:y_dim
        smoothed_errors(i,:) = smoothn(abs(y_pred(:,i) - Y_true(1:size(y_pred,1),i)), 'robust');
    end
    semilogy(smoothed_errors(:,1:end)');
    legend({'Absolute Error (Sine)'});
    title('Absolute Errors');
    
    %% plot log likelihood
    subplot(y_dim+2,1,y_dim+2);
    
    smoothed_logll = [];
    for i=1:y_dim
        smoothed_logll(i,:) = smoothn(log_marg_lik(:,i), 'robust');
    end
    plot(smoothed_logll');
    legend({'Log likelihood at test point'});
    title('Log Marginal Likelihood at test points');
    
    %% plot final 20% of results
    figure()
    y_dim = size(Y,2);
    for i=1:y_dim
        subplot(y_dim,1,i);
        st = ceil(0.8*N);
        errorbar(1:N-st+1, y_pred(st:N,i), sqrt(y_var(st:N)), 'bo'); hold on;
        scatter(1:N-st+1, Y_true(st:N,i), 'g+');
        ylim([-1.5, 1.5]);
        
        title(sprintf('Y(%d): Test Portion', i));
        hold off;
    end
    
    %% plot total time taken
    figure();
    semilogy(time_taken, '.-');
    xlabel('Time(t)');
    ylabel('Computation Time (s)');
    title('Computation Time per Iteration');
    
end

%% compute errors
tY = Y_true(trainEnd+1:end,1);
tpredY = y_pred(trainEnd+1:end,1);
tvarY = y_var(trainEnd+1:end,:);
%%
perf_rmse = rmse(tY, tpredY);
perf_nlpd = nlpd(tY, tpredY, tvarY);
%%
fprintf('Error Scores: %f (RMSE), %f (NLPD)\n', perf_rmse, perf_nlpd);

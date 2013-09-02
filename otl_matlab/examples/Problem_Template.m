%Problem_Template.m
%Author: Harold Soh
%This script sets up a OIESGP to run on a given problem. You can change the
%details here to match your problem. See the code comments.

%% Experimental parameters
clear(); 

do_update_plot = false; %show hyperparameters being updated?
do_final_plot = true;  %final plots
debug_mode = false; %leave this false usually

%% DATA COMES HERE

D = sin(0:0.5:2000*0.5)';
D = D + normrnd(0,0.05,size(D));
D = [D 5*rand(size(D,1), 2)];
X = D(1:end-1,:); %put your inputs here (one row per sample, columns are features)
Y = D(2:end,1);     %put your outputs here

x_dim = size(X,2);
y_dim = size(Y,2);
N = size(X,1);

%training for the first 80%
trainEnd = ceil(0.8*N);

%% MEMORY HERE 
memory_length = 10; %set to 1 if you don't want memory 
dl = FDelayLine(x_dim, memory_length);

%% Create the MemOGP (In this case the OIESGP)
%the most important parameters to change are the:
% 1. capacity: trades off accuracy and speed
% 2. ell: characteristic lengthscales
% 3. optimize_params: (true/false) do you want to optimise params?

%first specify our parametres
capacity = 100;       %basis vectors
epsilon = 1e-5;     %tolerance parameter

%we need to be careful while specifying the dimensions of the
%hyperparameters. This should be dimension of the MEMORY and not the input
kern_func = {'covRecSEardNoise'}; %This is the recursive ARD kernel

ell = zeros(x_dim,1) + log(5.0); %characteristic lengthscales (one for each dimension)
max_ell = zeros(x_dim,1) + log(1000000.0); 
min_ell = zeros(x_dim,1) + log(0.001);

sf2 = 0.5*log(0.5);             %signal variance parameter (log)
min_sf2 = 0.5*log(0.001);
max_sf2 = 0.5*log(10.0);

% optimisation parameters
optimize_params = true; %do you want optimisation
opt_interval = 10;
lambda0 = 0.01;
multiplier = 1.0;
gradient_bound = 0.1;
pm_stop_opt_thres = 10;

tau = memory_length;

rho = 0.9;
min_rho = 0.5;
max_rho = 1.1;

noise = 0.01;
lognoise = 0.5*log(noise); %noise hyperparameter
min_lognoise = 0.5*log(0.001);
max_lognoise = 0.5*log(0.5);


kern_params = [ell; sf2; rho; lognoise];    %kernel parameters
lower_bounds = [min_ell; min_sf2; min_rho; min_lognoise];
upper_bounds = [max_ell; max_sf2; max_rho; max_lognoise];
%kern_params = [ell; sf2; lognoise];

rfilt = 0.8; %filter for recency scores
sw = 1.0;      %weights between recency and reconstruction
regularizer = 1.0; %regularizer for p and q


pm_hyp_to_optimise = 1:length(kern_params); % which parameters to optimise
pm_update_thres = 0.01;
pm_num_opt_per_itr = length(kern_params);


n_kern_params = length(kern_params);


% error 'e' or likelihood 'p' based optimisation
pm_lik_mode = 'p'; 
noisefunc = @getRecKernNoise;

gp_params = [capacity, 0.0, epsilon, rfilt, sw, regularizer];
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


%% training for the first 80%
for i=1:trainEnd %size(X,1)

   
    nbvs = pMemOGP.getNumBasisVectors();
    fprintf('Iteration %d, num BVs: %d\n', i, nbvs);
    tic;
    pMemOGP.updateMemory(X(i,:));
    [y_pred(i,:), y_var(i,:), log_marg_lik(i,:), MemOGP_rebuilt] = pMemOGP.trainInterval(Y(i,:));
    time_taken(i) = toc();
    
    all_nbvs(i) = nbvs;
    
    kern_params = pMemOGP.getKernParams();

end

%% testing for the final 20%
for i=trainEnd+1:N %size(X,1)
    fprintf('Iteration %d\n', i);
    tic;
    pMemOGP.updateMemory(X(i,:));
    [y_pred(i,:), y_var(i,:)] = pMemOGP.predict();
    time_taken(i) = toc();
    
end

%% Plot each dimension in its own plot
if do_final_plot
    figure()
    NN = size(y_pred,1);
    y_dim = size(Y,2);
    
    for i=1:y_dim
        subplot(y_dim+2,1,i);
        errorbar(1:NN, y_pred(1:NN,i), y_var(1:NN), 'b-'); hold on;
        scatter(1:size(y_pred,1), Y(1:size(y_pred,1),i), 'g+');
        ylim([-1.5, 1.5]);
        title(sprintf('Y(%d): All Data', i));
        hold off;
    end
    
    %% plot the absolute error
    subplot(y_dim+2,1,y_dim+1);
    
    for i=1:y_dim
        smoothed_errors(i,:) = smoothn(abs(y_pred(:,i) - Y(1:size(y_pred,1),i)), 'robust');
    end
    semilogy(smoothed_errors(:,1:end)');
    legend({'Absolute Error (Sine)'});
    title('Absolute Errors');
    
    %% plot time taken
    subplot(y_dim+2,1,y_dim+2);
    semilogy(time_taken, '.-');
    xlabel('Time(t)');
    ylabel('Computation Time (s)');
    title('Computation Time per Iteration');
    
    %% plot final 20% of results
    figure()
    y_dim = size(Y,2);
    for i=1:y_dim
        subplot(y_dim,1,i);
        st = ceil(0.8*N);
        errorbar(1:NN-st+1, y_pred(st:NN,i), sqrt(y_var(st:NN)), 'bo'); hold on;
        scatter(1:NN-st+1, Y(st:NN,i), 'g+');
        ylim([-1.5, 1.5]);
        
        title(sprintf('Y(%d): Test Portion', i));
        hold off;
    end
    

    
end

%% compute errors
tY = Y(trainEnd+1:end,1);
tpredY = y_pred(trainEnd+1:end,1);
tvarY = y_var(trainEnd+1:end,:);
%%
perf_rmse = rmse(tY, tpredY);
perf_nlpd = nlpd(tY, tpredY, tvarY);
%%
fprintf('Error Scores: %f (RMSE), %f (NLPD)\n', perf_rmse, perf_nlpd);

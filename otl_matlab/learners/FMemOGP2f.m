function [pmgp] = FMemOGP2f(inf_mode, params, mem_obj, ...
    kern_func, kern_params, kern_noise_func, ...
    opt_params, ...
    deletion_criteria, debug_mode)


%to copy or not to copy, that is the question
if nargin == 1
    rhs = inf_mode;
    if isstruct(rhs)
        if strcmp(rhs.ftype, 'FMemOGP2f')
            %disp('Copying FMemOGP2e!');
            setInternals(rhs.getInternals());
        else
            error('Non-matching FMemOGP2f type. Cannot copy!');
        end
        
    else
        help FMemOGP2f;
    end
else
    
    pm_kern_noise_func = kern_noise_func;

    ftype = 'FMemOGP2f';
    %declare all the variables to start
    pm_inf_mode = 0;           %inference mode
    pm_capacity = 0;           %how many basis vectors to store
    pm_noise = 0;              %noise prior
    pm_epsilon = 0;            %tolerance parameter (set to 1e-6, increase if
    %numerical errors are apparent
    pm_gp_params = 0;          %backup of parameters
    
    pm_covf = 0;               %covariance function
    pm_covf_params = 0;        %covariance function parameters
    
    
    %internal parameters
    pm_alpha = 0;
    pm_C = 0;                  %covariance matrix
    pm_Q = 0;                  %inverse of covariance matrix
    pm_hyp_gradients = 0;      %gradients for hyperparameters
    
    pm_hyp_to_optimise = [];   %parameters to optimise
    pm_num_opt_per_itr = [];   %number of hyperparameters to optimise per iteration
    pm_hyp_cell = {};          %cell array of hyperparameters
    
    pm_deletion_criteria = 0;  %deletion criteria for basis vectors once
    %capacity is reached. Options are 'm' for
    %minimax and 'n' for 2-norm
    
    pm_basis_vectors = 0;      %stored basis vectors
    pm_n_bvs = 0;              %number of basis vectors
    pm_ys = 0;                 %the stored y values
    
    pm_memory = 0;             %the internal memory
    
    pm_optimize_params = 0;    %bool indiciating if we optimize parameters
    pm_optAlg = 0;             %optimisation method
    pm_opt_interval = 0;       %interval before we re-optimise the MemOGP2
    pm_t = 0;                  %internal time counter
    pm_last_opt_time = 0;      %when was model was optimized?
    pm_not_rebuilt_count = 0;
    pm_lik_mode = 'l';
    pm_debug_mode = 0;         %set to true to get some warnings
    %internal anonymous functions
    pm_rowNorm = 0;
    
    pm_gradients = 0;
    
    pm_recency_scores = 0;     %recency scores capture how often a bv is used
    pm_rfilt = 0;              %filter for recency scores (0-1.0)
    pm_sw = 0;                 %weights for score preference (0-1.0) 1.0 means total weight towards reconstruction vs. recency
    
    pm_regularizer = 0;        %regularizer for q and r computation (prevents numerical errors with over/underflows
    %0 means zero weight for score and full weight
    %to recency effect
    pm_covf_params = kern_params;
    minvar = kern_noise_func(pm_covf_params);
    pm_stop_opt_thres = Inf;
    bv_mean = [];
    pm_dK = {};
    pm_dK_valid = zeros(size(pm_covf_params));
    train_x_buffer = [];
    train_y_buffer = [];
    
    pm_KK = []; %feval(pm_covf{:},pm_covf_params, X, X);
    pm_UK = []; %triu(kern_mat);
    pm_kyz = 0; %mean(K(UK ~= 0));
    
    %set defaults
    if nargin < 7
        deletion_criteria = 'n';
        debug_mode = false;
    end
    if nargin < 8
        debug_mode = false;
    end
    
    %initialise
    if any(inf_mode == ['r','c'])
        pm_inf_mode = inf_mode;
    else
        error('Inference mode parameter (inf_mode) must be (r)egression or (c)lassification');
    end
    
    
    %initialise the pmgp
    pm_init(params, kern_func, kern_params, ...
        deletion_criteria, debug_mode);
    
    %set up the optimiser
    pm_opt_params = opt_params;
    if isempty(opt_params)
        pm_optimize_params = false;
    else
        conv_opt_params = cell2mat(opt_params(1:3));
        pm_opt_interval = int32(round(conv_opt_params(1)));
        lambda0 = conv_opt_params(2);
        multiplier = conv_opt_params(3);
        gradient_bound = cell2mat(opt_params(4));
        
        if length(opt_params) < 7
            pm_optAlg = FOHGD(kern_params, lambda0, multiplier, ...
                gradient_bound, 'max');
        elseif length(opt_params) < 8
            plotfunc = cell2mat(opt_params(7));
            auto_plot = false;
            pm_optAlg = FOHGD(kern_params, lambda0, multiplier, ...
                gradient_bound, 'max', plotfunc, auto_plot);
        else
            plotfunc = cell2mat(opt_params(7));
            auto_plot = cell2mat(opt_params(8));
            pm_optAlg = FOHGD(kern_params, lambda0, multiplier, ...
                gradient_bound, 'max', plotfunc, auto_plot);
        end
        
        
        
        if length(opt_params) >= 5
            pm_optAlg.setLowerBounds(cell2mat(opt_params(5)));
        end
        
        if length(opt_params) >= 6
            pm_optAlg.setUpperBounds(cell2mat(opt_params(6)));
        end
        
        %by default we optimise everything and all elements per iteration
        pm_hyp_to_optimise = [1:length(kern_params)];   
        pm_num_opt_per_itr = length(kern_params);
        
        if length(opt_params) >=7
            pm_hyp_to_optimise = cell2mat(opt_params(9));   
            pm_num_opt_per_itr = cell2mat(opt_params(10));
            pm_update_thres = cell2mat(opt_params(11));
            pm_stop_opt_thres = cell2mat(opt_params(12));
            pm_lik_mode = cell2mat(opt_params(13));
        end
        
        pm_hyp_cell = vecToCellPartition(pm_hyp_to_optimise,pm_num_opt_per_itr);
    end
    
    %set up the memory
    pm_memory = mem_obj;
    
    %set up useful helpder function
    pm_rowNorm = @(X,P) sum(abs(X).^P,2).^(1/P);
    
    %initialise parameters that we want set up only once
    pm_hyp_gradients = zeros(size(pm_covf_params));
    pm_t = 1;
    pm_last_opt_time = 0;
    
end

%function initialisations
pmgp.ftype = ftype;
pmgp.init = @pm_init;
pmgp.optimizeModel = @pm_optimizeModel;
pmgp.updateMemory = @pm_updateMemory;
pmgp.resetMemory = @pm_resetMemory;
pmgp.getMemory = @pm_getMemory;
pmgp.setMemory = @pm_setMemory;
pmgp.Erf = @pm_Erf;
pmgp.train = @pm_train;
pmgp.trainInterval = @pm_trainInterval;
pmgp.predict = @pm_predict;
pmgp.deleteBasisVector = @pm_deleteBasisVector;
pmgp.getHypGradients = @pm_getHypGradients;
pmgp.setHypOpt = @pm_setHypOpt;
pmgp.getHypOpt = @pm_getHypOpt;
pmgp.getKernParams = @pm_getKernParams;
pmgp.setKernParams = @pm_setKernParams;
pmgp.getKStar = @pm_getKStar;
pmgp.getNumBasisVectors = @pm_getNumBasisVectors;
pmgp.getBasisVectors = @pm_getBasisVectors;
pmgp.getKernNoise = kern_noise_func;
pmgp.getMeanBasisVector = @pm_getMeanBasisVector;
pmgp.getDistanceScore = @pm_getKernAtMeanBasisVector;
pmgp.getRelativeDistanceScore = @pm_getRelKernAtMeanBasisVector;
pmgp.getKernelDistanceScore = @pm_getKernelDistanceScore;
pmgp.getNoise = @pm_getNoise;

pmgp.getInternals = @getInternals;
pmgp.setInternals = @setInternals;
pmgp.reset = @reset;



    function reset(optimize_params)
        pm_alpha = 0;
        pm_C = 0;                  %covariance matrix
        pm_Q = 0;                  %inverse of covariance matrix
        pm_hyp_gradients = 0;      %gradients for hyperparameters
        
        pm_n_bvs = 0;
        
        pm_basis_vectors = [];
        pm_ys = [];
        bv_mean = [];
        pm_recency_scores = 0;
        train_x_buffer = [];
        train_y_buffer = [];
        pm_t = 0; %rhs.pm_t;
        pm_last_opt_time = 0;
        
        pm_KK = []; %feval(pm_covf{:},pm_covf_params, X, X);
        pm_UK = []; %triu(kern_mat);
        pm_kyz = 0; %mean(K(UK ~= 0));
        pm_optimize_params = optimize_params;
        pm_dK = {};
        pm_dK_valid = zeros(size(pm_covf_params));
        %pm_optAlg.resetOpt();
        
    end


    function [noise] = pm_getNoise()
        noise = kern_noise_func(pm_covf_params);
       
    end



    function [ost] = getInternals()
        ost.ftype = ftype;
        ost.kern_noise_func = kern_noise_func;
        ost.pm_inf_mode = pm_inf_mode;
        ost.pm_capacity = pm_capacity;
        ost.pm_noise = pm_noise;
        ost.pm_epsilon = pm_epsilon;
        ost.pm_gp_params = pm_gp_params;
        ost.pm_covf = pm_covf;
        ost.pm_covf_params = pm_covf_params;
        ost.pm_alpha = pm_alpha;
        ost.pm_C = pm_C;
        ost.pm_Q = pm_Q;
        ost.pm_hyp_gradients = pm_hyp_gradients;
        ost.pm_deletion_criteria = pm_deletion_criteria;
        ost.pm_basis_vectors = pm_basis_vectors;
        ost.pm_n_bvs = pm_n_bvs;
        ost.pm_ys = pm_ys;
        ost.pm_memory = pm_memory;
        ost.pm_optimize_params = pm_optimize_params;
        ost.pm_optAlg = pm_optAlg;
        ost.pm_opt_interval = pm_opt_interval;
        ost.pm_t = pm_t;
        ost.pm_last_opt_time = pm_last_opt_time;
        ost.pm_debug_mode = pm_debug_mode;
        ost.pm_rowNorm = pm_rowNorm;
        ost.pm_gradients = pm_gradients;
        ost.pm_recency_scores = pm_recency_scores;
        ost.pm_rfilt = pm_rfilt;
        ost.pm_sw = pm_sw;
        ost.pm_regularizer = pm_regularizer;
        ost.minvar = minvar;
        ost.bv_mean = bv_mean;
        ost.debug_mode = debug_mode;

        ost.pm_opt_params = pm_opt_params;

        ost.train_x_buffer = train_x_buffer;
        ost.train_y_buffer = train_y_buffer;
        
        ost.pm_KK = pm_KK; %feval(pm_covf{:},pm_covf_params, X, X);
        ost.pm_UK = pm_UK; %triu(kern_mat);
        ost.pm_kyz = pm_kyz; %mean(K(UK ~= 0));
        
        ost.pm_hyp_to_optimise = pm_hyp_to_optimise;   
        ost.pm_num_opt_per_itr = pm_num_opt_per_itr;   
        ost.pm_hyp_cell = pm_hyp_cell;
        ost.pm_not_rebuilt_count = pm_not_rebuilt_count;
        
        ost.pm_update_thres = pm_update_thres;
        ost.pm_stop_opt_thres = pm_stop_opt_thres;
        ost.pm_lik_mode = pm_lik_mode;
        
        ost.pm_dK = pm_dK;
        ost.pm_dK_valid = pm_dK_valid;
    end

    function setInternals(rhs)
        ftype = rhs.ftype;
        kern_noise_func = rhs.kern_noise_func;
        pm_optAlg = FOHGD(rhs.pm_optAlg); %this is special
        pm_t = rhs.pm_t;
        pm_last_opt_time = rhs.pm_last_opt_time;
        
        
        pm_inf_mode = rhs.pm_inf_mode;
        pm_capacity = rhs.pm_capacity;
        pm_noise = rhs.pm_noise;
        pm_epsilon = rhs.pm_epsilon;
        pm_gp_params = rhs.pm_gp_params;
        pm_covf = rhs.pm_covf;
        pm_covf_params = rhs.pm_covf_params;
        pm_alpha = rhs.pm_alpha;
        pm_C = rhs.pm_C;
        pm_Q = rhs.pm_Q;
        pm_hyp_gradients = rhs.pm_hyp_gradients;
        pm_deletion_criteria = rhs.pm_deletion_criteria;
        pm_basis_vectors = rhs.pm_basis_vectors;
        pm_n_bvs = rhs.pm_n_bvs;
        pm_ys = rhs.pm_ys;
        pm_memory = rhs.pm_memory;
        pm_optimize_params = rhs.pm_optimize_params;
        
        pm_opt_interval = rhs.pm_opt_interval;

        pm_debug_mode = rhs.pm_debug_mode;
        pm_rowNorm = rhs.pm_rowNorm;
        pm_gradients = rhs.pm_gradients;
        pm_recency_scores = rhs.pm_recency_scores;
        pm_rfilt = rhs.pm_rfilt;
        pm_sw = rhs.pm_sw;
        pm_regularizer = rhs.pm_regularizer;
        minvar = rhs.minvar;
        bv_mean = rhs.bv_mean;
        debug_mode = rhs.debug_mode;

        pm_opt_params = rhs.pm_opt_params;

        train_x_buffer = rhs.train_x_buffer;
        train_y_buffer = rhs.train_y_buffer;
        
        pm_KK = rhs.pm_KK; %feval(pm_covf{:},pm_covf_params, X, X);
        pm_UK = rhs.pm_UK; %triu(kern_mat);
        pm_kyz = rhs.pm_kyz; %mean(K(UK ~= 0));
        
        pm_hyp_to_optimise = rhs.pm_hyp_to_optimise;   
        pm_num_opt_per_itr = rhs.pm_num_opt_per_itr;   
        
        pm_hyp_cell = rhs.pm_hyp_cell;
        pm_not_rebuilt_count = 0;
        pm_update_thres = rhs.pm_update_thres;
        pm_stop_opt_thres = rhs.pm_stop_opt_thres;
        
        pm_lik_mode = rhs.pm_lik_mode;
        
        pm_dK = rhs.pm_dK;
        pm_dK_valid = rhs.pm_dK_valid;
    end


    function k = pm_getKernAtMeanBasisVector(x)
        if pm_n_bvs == 0
            k = 0;
            return;
        end
        if nargin == 0
            x = pm_memory.getState();
        end
        k = feval(pm_covf{:},pm_covf_params, bv_mean, x);
    end

    function kstar = pm_getKStar(x)
        if pm_n_bvs == 0
            k = 0;
            return;
        end
        if nargin == 0
            x = pm_memory.getState();
        end
        kstar = feval(pm_covf{:},pm_covf_params, x);
    end



    function k = pm_getRelKernAtMeanBasisVector(x)
        if pm_n_bvs == 0
            k = 0;
            return;
        end
        if nargin == 0
            x = pm_memory.getState();
        end
        k = feval(pm_covf{:},pm_covf_params, bv_mean, x);
        kstar = feval(pm_covf{:},pm_covf_params, x, x) - kern_noise_func(pm_covf_params);
        k = k/kstar;
    end

    function dk = pm_getKernelDistanceScore(x)

        if nargin == 0
            x = pm_memory.getState();
        end
        
        if pm_n_bvs == 0
            dk = 0;
            return;
        end
        X = pm_basis_vectors(1:pm_n_bvs,:);
        kxx = feval(pm_covf{:},pm_covf_params, x) - kern_noise_func(pm_covf_params);
        kxy = mean(feval(pm_covf{:},pm_covf_params, X, x));
        
        dk = kxx -2*kxy + pm_kyz;
        %dk = dk/(kxx + pm_kyz);
        
        dk = dk/kxx;
    end


    function n_bvs = pm_getNumBasisVectors()
        n_bvs = pm_n_bvs;
    end

    function [bvs, ys] = pm_getBasisVectors()
        bvs = pm_basis_vectors;      %stored basis vectors
        ys = pm_ys;                 %the stored y values
    end

    function [mbv] = pm_getMeanBasisVector()
        mnv = bv_mean;
    end


    function memstate = pm_getMemory()
        memstate = pm_memory.getState();
    end

    function pm_setMemory(memstate)
        pm_memory.setState(memstate);
    end

    function pm_init(params, kern_func, kern_params, ...
            deletion_criteria, debug_mode)
        %INIT Initialises the MemOGP2
        pm_gp_params = params;
        pm_capacity = params(1);
        pm_noise = params(2);
        
        if length(params) >= 3
            pm_epsilon = params(3);
            pm_rfilt = params(4); %filter for recency scores
            pm_sw = params(5);      %weights between recency and reconstruction
            pm_regularizer = params(6); %regularizer for p and q
        else
            pm_epsilon = 1e-6;
            pm_rfilt = 0.8; %filter for recency scores
            pm_sw = 1.0;      %weights between recency and reconstruction
            pm_regularizer = 1.0; %regularizer for p and q
        end
        
        pm_deletion_criteria = deletion_criteria;
        pm_debug_mode = debug_mode;
        
        pm_covf = kern_func;
        pm_covf_params = kern_params;
        pm_dK_valid = zeros(size(pm_covf_params));
        
        %initialisations
        pm_alpha = 0;
        pm_C = 0;
        
        pm_n_bvs = 0;
        
        pm_basis_vectors = [];
        pm_ys = [];
        
        pm_KK = []; 
        pm_UK = []; 
        pm_kyz = 0;
        pm_t = 0;
    end


    function pm_rebuild(kern_func, kern_params)
        %REBUILDMemOGP2 Rebuilds the MemOGP2 from basis vector set.
        %allows you to change the parameters and kernel function/params
        
        %to "jumpstart" the MemOGP2 after a parameter change, we go
        %through the stored basis vectors and ys to rebuild the MemOGP2
        
        
        %save the basis vector set and ys
        if not(isempty(kern_func))
            pm_covf = kern_func;
        end
        
        pm_covf_params = kern_params;
        minvar = kern_noise_func(pm_covf_params);
        %re-initialise the MemOGP2
        readd = false;
        
        %uncomment the next few lines if you want to start "fresh"
        %pm_init(pm_gp_params, kern_func, pm_covf_params, ...
        %    pm_deletion_criteria, pm_debug_mode);
        %return
        
        if readd
            
            stored_bvs = pm_basis_vectors(1:pm_n_bvs, :);
            n_bvs = pm_n_bvs;
            stored_ys = pm_ys;
            
            pm_n_bvs = 0;
            
            pm_optimize_params = false;
            for j=1:n_bvs
                pm_train(stored_bvs(j,:), stored_ys(j,:));
            end
            
            pm_optimize_params = true;
            
        else
            %recompute C, Q and alpha
            pm_C = feval(pm_covf{:},pm_covf_params, pm_basis_vectors(1:pm_n_bvs, :));
            pm_KK = pm_C - eye(size(pm_C))*kern_noise_func(pm_covf_params);
            pm_UK = triu(pm_KK);
            pm_kyz = mean(pm_UK(pm_UK ~= 0));
            
            pm_C = -inv(pm_C + eye(size(pm_KK))*5e-6);
            pm_Q = inv(pm_KK + eye(size(pm_KK))*5e-6);
            pm_alpha = -pm_C*pm_ys(1:pm_n_bvs,:);
            
        end
        
        if pm_debug_mode
            fprintf('Info: model rebuilt\n');
            pm_covf_params
        end
        
    end

    function [updated] = pm_optimizeModel()
        
        if pm_optimize_params == false
            return;
        end
        
        
        %         if pm_inf_mode == 'c'
        %             disp('Optimising classification hyperparameters');
        %         else
        %             disp('Optimising regression hyperparameters');
        %         end
        
        [new_kern_params, ~] = pm_optAlg.getParams();
        if isempty(new_kern_params)
            updated = false;
            return;
        end
        
        score = (norm(new_kern_params(pm_hyp_to_optimise) - pm_covf_params(pm_hyp_to_optimise)))^2;
        norm_score = score; %/length(pm_hyp_to_optimise); %/norm(pm_covf_params(pm_hyp_to_optimise));
        
        if norm_score > pm_update_thres 
            %we change the hyperparameters and rebuild the MemOGP2
            %disp('Optimising');
            
            [new_kern_params] = pm_optAlg.updateParams(true, true);
            pm_rebuild(pm_covf, new_kern_params);
            
            pm_not_rebuilt_count = 0;
            updated = true;
        else
            %disp('Not optimising');
             [new_kern_params] = pm_optAlg.updateParams(true, true);
             pm_rebuild(pm_covf, new_kern_params);
            
            updated = true;
            pm_not_rebuilt_count = pm_not_rebuilt_count+1;
        end
        
        if pm_not_rebuilt_count > pm_stop_opt_thres
           disp('no longer optimising');
           pm_optimize_params = false;
        end
        
        %pm_dK = {};
        pm_dK_valid = zeros(size(pm_covf_params));
        pm_last_opt_time = pm_t;
        pm_hyp_gradients = zeros(size(pm_covf_params));
    end


    function [hyp_gradients] = pm_getHypGradients(y, th_params)
        %GETHYPGRADIENTS Get the hyperparameter gradients.
        %
        x = pm_memory.getState();
        y_dim = size(y,2);
        
        %Check if this MemOGP2 is empty
        if pm_n_bvs == 0
            hyp_gradients = zeros(y_dim,size(pm_covf_params,1));
            return;
        end
        
        %perform gradient computations
        alpha = pm_alpha(1:pm_n_bvs,:);
        X = pm_basis_vectors(1:pm_n_bvs,:);
        
        ks = feval(pm_covf{:},pm_covf_params, x);
        Kss = feval(pm_covf{:},pm_covf_params, X, x);
        
        %Kinv = pm_Q(1:pm_n_bvs,1:pm_n_bvs);
        Kinv = -pm_C(1:pm_n_bvs,1:pm_n_bvs);
        KinvKss = Kinv*Kss;
        
        u = Kss'*alpha;
        s2 = ks - Kss'*KinvKss;

        
        %for classification compute differently
        if pm_inf_mode == 'c'
            if s2 < 0
                s2 = 0.001;
            end
            
            sx = sqrt(s2);
            pred_Erfz = pm_Erf(u/sx);
            
            if pred_Erfz > 0.5
                pred_mean = 1;
            else
                pred_mean = -1;
            end
            pred_var = pred_Erfz;
            z0 = u/sx;
            z = y.*z0;
            
            Erfz = pm_Erf(z);
            dErfz = 2.0/sqrt(2*pi)*exp(-(z.^2)/2);
            
            if abs(dErfz) < 1e-20
                dErfz_ratio = 0.0;
            else
                dErfz_ratio = (dErfz./Erfz);
            end
            
        end
        
        
        %for each output
        dpy_dth = zeros(y_dim, size(pm_covf_params,1));
        n_params = length(pm_covf_params);
        
        if nargin <= 1
            th_params = [1:n_params];
        end
        
       

        for tt=1:length(th_params);
            th = th_params(tt);

            dKss =  feval(pm_covf{:},pm_covf_params, X, x, th);
            
            dks = feval(pm_covf{:},pm_covf_params, x, [], th);


                       
            if pm_dK_valid(th) == 0
                pm_dK{th} = feval(pm_covf{:},pm_covf_params, X, X, th);
                pm_dK_valid(th) = 1;
            end
            
            if size(alpha,1) ~= size(pm_dK{th},1)
                pm_dK{th} = feval(pm_covf{:},pm_covf_params, X, X, th);
                pm_dK_valid(th) = 1;
            end
            
            for i=1:y_dim
                %get the gradients with respect to each hyperparameter
                if pm_inf_mode == 'c'
                    dp_du = y./sx .* dErfz_ratio;
                    dp_ds2 = -u*dp_du/sx;
                elseif pm_inf_mode == 'r'
                    diff_y = (y(i) - u(i));
                    diff_y2 = diff_y.^2;
                    
                    if pm_lik_mode == 'e'
                        dp_du = 2*diff_y;
                        dp_ds2 = 0;
                    elseif pm_lik_mode == 'p'
                        dp_du = diff_y/s2;
                        dp_ds2 = diff_y2/(2*(s2.^2)) - 1/(2*s2);
                    else
                        error('No such likelihood model');
                    end
                else
                    error('Internal error: pm_inf_mode must be set to (r)egression or (c)lassification');
                end
                
                %size(dKss)
                %size(alpha)
                %size(pm_dK{th})
                du_dth = dKss'*alpha(:,i) +  Kss'*Kinv*pm_dK{th}*alpha(:,i);
                %du_dth = dKss'*alpha(:,i) +  Kss'*Kinv*dK*alpha(:,i);
                ds2_dth = 0;
                if pm_lik_mode == 'p'
                    ds2_dth = dks - ...
                        (dKss'*KinvKss - ...
                        Kss'*(Kinv*(pm_dK{th}*KinvKss)) + ...
                        Kss'*(Kinv*dKss));


                end
                
                dpy_dth(i,th) = dp_du*du_dth + dp_ds2*ds2_dth;
                
            end
        end
        
        hyp_gradients = dpy_dth;
        
    end




    function [pred_mean, pred_var, log_marg_likl, rebuilt] = pm_trainInterval(x_in, y_in)
                %TRAIN Trains the MemOGP2 using input x and output y
        if nargin == 2
            x = x_in;
            y = y_in;
        elseif nargin == 1
            x = pm_memory.getState();
            y = x_in;
        else
            error('Error using Training function');
            return;
        end
        rebuilt = false;

        n_params = length(pm_covf_params);
        
        train_x_buffer(end+1,:) = x;
        train_y_buffer(end+1,:) = y;
        
        if pm_optimize_params && (pm_n_bvs > 0) && (pm_t > 2*pm_capacity) %~= 0 %heuristic, only start optimising if we have some elements
            %check if we are supposed to optimise hyperparameters
            %if and(pm_optimize_params ,pm_t > pm_capacity + 1)
            if pm_optimize_params
                %yes, let's check the current "optimization time"

                %compute the gradients
                th = mod(pm_t, length(pm_hyp_cell)) + 1;
                
                [gradients] = pm_getHypGradients(y, pm_hyp_cell{th});

                if isempty(pm_gradients)
                    pm_gradients = gradients;
                else
                    pm_gradients = pm_gradients + gradients;
                end


                if th == length(pm_hyp_cell) %finished updating all gradients

                        if length(y) > 1
                            gradients = nansum(pm_gradients)';
                        else
                            gradients = pm_gradients;
                            gradients(isnan(pm_gradients)) = 0;
                            gradients = gradients';
                        end


                    %pm_gradients
                    %gradients
                    %update the mean of the hyp_gradients
                    if any(isnan(gradients))
                        warning('Warning! Gradient is NaN! Not including in this update');
                    else
                        pm_optAlg.update(gradients);
                        pm_gradients = [];
                    end
                end

                %perform whole model optimization if necessary
                if pm_opt_interval > 0
                    if mod(pm_t, length(pm_hyp_cell)*pm_opt_interval) == 0
                        pm_optimizeModel();
                        pm_gradients = [];
                        pm_hyp_cell = pm_hyp_cell(randperm(length(pm_hyp_cell)));
                        rebuilt = true;
                    end
                end

                
            end
            
            if not(rebuilt)                
                kstar = feval(pm_covf{:},pm_covf_params,x);

                %end
                y_dim = size(y,2);


                k = feval(pm_covf{:},pm_covf_params, pm_basis_vectors(1:pm_n_bvs, :), x);
                m = k'*pm_alpha(1:pm_n_bvs, :);

                if any(isnan(m))
                    x
                    kernel_params = pm_covf_params
                    k
                    pm_alpha
                    error('MemOGP2: Training. Predicted mean is NaN');
                end

                Ck = pm_C(1:pm_n_bvs, 1:pm_n_bvs)*k;
                s2 = kstar + (k.'*Ck);
                noiseval = kern_noise_func(pm_covf_params);
                if (s2 < noiseval)
                    
                    if (pm_debug_mode)
                        fprintf(1, 'WARNING: variance too small. Possible numerical error. Try increasing tolerance?: %f\n',s2);
                        kstar
                        norm(Ck)
                    
                    end
                    
                    
                    s2 = minvar;
                end


                pred_mean = m;
                pred_var = s2;
                pred_var = zeros(1,y_dim) + pred_var;

                log_marg_likl = log(normpdf(y, m, sqrt(s2)));
            end
           
            
        else
            rebuilt = true;
        end


        if rebuilt
            for xi = randperm(size(train_x_buffer,1))
                x = train_x_buffer(xi,:);
                y = train_y_buffer(xi,:);
                
                
                %pm_t = pm_t + 1;
                %continue on with training
                %if not(pm_optimize_params)
                
                kstar = feval(pm_covf{:},pm_covf_params,x);
                
                %end
                y_dim = size(y,2);
                
                
                %is this a new pmgp?
                if pm_n_bvs == 0
                    pm_alpha = y / (kstar) ;
                    pm_ys = y;
                    pm_KK = kstar;
                    pm_UK = kstar;
                    pm_kyz = kstar;
                    pm_C = -1/(kstar); %+ pm_noise);
                    noiseval = kern_noise_func(pm_covf_params);
                    pm_Q = 1/(kstar - noiseval);
                    
                    pm_KK = kstar - noiseval;
                    pm_UK = pm_KK;
                    pm_kyz = pm_KK;
                    
                    pm_basis_vectors(1,:) = x;
                    bv_mean = x;
                    pm_n_bvs = 1;
                    pred_mean = zeros(1, y_dim);
                    pred_var = zeros(1, y_dim);
                    log_marg_likl = log(normpdf(y, 0, kstar));
                    pm_recency_scores = [kstar];
                    pm_t = 0;                   
                else
                    
                    %nope, let's do a geometric test
                    
                    %if not(pm_optimize_params)
                    k = feval(pm_covf{:},pm_covf_params, pm_basis_vectors(1:pm_n_bvs, :), x);
                    m = k'*pm_alpha(1:pm_n_bvs, :);
                    
                    if any(isnan(m))
                        x
                        kernel_params = pm_covf_params
                        k
                        pm_alpha
                        error('MemOGP2: Training. Predicted mean is NaN');
                    end
                    
                    Ck = pm_C(1:pm_n_bvs, 1:pm_n_bvs)*k;
                    s2 = kstar + (k.'*Ck);
                    
                    noiseval = kern_noise_func(pm_covf_params); 
                    
                    if (s2 < noiseval)
                        if (pm_debug_mode)
                            fprintf(1, 'WARNING: variance too small. Possible numerical error. Try increasing tolerance?: %f\n',s2);
                        end
                        s2 = noiseval;
                    end
                    
                    pred_mean = m;
                    pred_var = s2;

                    pred_var = zeros(1,y_dim) + pred_var;
                    
                    if nargout >= 3
                        log_marg_likl = log(normpdf(y, m, sqrt(s2)));
                    end
                    
                    
                    if pm_inf_mode == 'r'
                        r = -1.0/(s2);
                        q = -r*(y - m);
                    elseif pm_inf_mode == 'c'
                        
                        %                     if s2 < noise
                        %                         s2 = noise;
                        %                     end
                        
                        sx = sqrt(s2);
                        pred_Erfz = pm_Erf(m/sx);
                        
                        if pred_Erfz > 0.5
                            pred_mean = 1;
                        else
                            pred_mean = -1;
                        end
                        pred_var = pred_Erfz;
                        z0 = m/sx;
                        z = y.*z0;
                        
                        Erfz = pm_Erf(z);
                        if abs(Erfz) < 1e-20
                            q = 0.0;
                            r = 0.0;
                        else
                            dErfz = pm_regularizer/sqrt(2*pi)*exp(-(z.^2)/2);
                            dErfz2 = pm_regularizer*dErfz.*(-z);
                            
                            if abs(dErfz) < 1e-20
                                q = 0.0;
                                r = 0.0;
                            else
                                q = y./sx .* (dErfz./Erfz);
                                r = (1/s2)*(dErfz2./dErfz - (dErfz./Erfz)^2);
                            end
                        end
                    else
                        error('Error in training. Inference mode must be either (r)egression or (c)lassfication');
                    end
                    
                    
                    
                    
                    if isnan(r) || isinf(r)
                        r
                        q
                        y
                        m
                        s2
                        error('r is NaN');
                    end
                    
                    
                    ehat = pm_Q(1:pm_n_bvs,1:pm_n_bvs)*k;
                    noiseval = kern_noise_func(pm_covf_params);
                    gamma = kstar - dot(k,ehat) - noiseval; %modified because the kernel now includes the noise term
                    
                    eta = 1.0./(1.0 + gamma*r);
                    
                    if or ( any(abs(q) > 1e3), any(abs(r) > 1e3) )
%                         m
%                         s2
%                         ehat
%                         eta
%                         gamma
                         %warning('q or r seems too large. Skipping this update');
                        continue;
                    end
                    
                    if (gamma < -1e-4 )
                        if (pm_debug_mode)
                            fprintf(1, 'WARNING: gamma too small. Possible numerical issue. Try increasing tolerance?: %f\n', gamma);
                        end
                        %warning('gamma is too small');
                        
                        
                        gamma = 0;
                        %don't process because of possible numerical issue
                        %continue;
                    end
                    %pm_epsilon*(kstar-noiseval)
                    if (gamma >= pm_epsilon) %*(kstar-noiseval)) %pm_epsilon*sig_var*noiseval) %pm_epsilon) %pm_epsilon*sig_var)%pm_epsilon*sig_var*noiseval) %pm_epsilon*(kstar-noiseval)) %
                        %full update
                        if pm_debug_mode
                            disp('Info: full update');
                        end
                        s = [Ck; 1];
                        
                        
                        %add to bv
                        
                        pm_n_bvs = pm_n_bvs+1;
                        bv_mean = (1/pm_n_bvs)*x + ((pm_n_bvs-1)/pm_n_bvs)*bv_mean;
                        
                        pm_basis_vectors(pm_n_bvs,:) = x;
                        
                        pm_ys(pm_n_bvs,:) = y;
                        
                        %update Q
                        pm_Q(pm_n_bvs, 1:pm_n_bvs) = 0;
                        pm_Q(1:pm_n_bvs, pm_n_bvs) = 0;
                        
                        ehat(pm_n_bvs) = -1;
                        ehat = ehat(:); %make ehat a column vector
                        
                        pm_Q(1:pm_n_bvs, 1:pm_n_bvs) = pm_Q(1:pm_n_bvs, 1:pm_n_bvs) + (1/gamma)*(ehat*ehat.');
                        
                        pm_alpha(pm_n_bvs,:) = 0;
                        pm_alpha(1:pm_n_bvs,:) = pm_alpha(1:pm_n_bvs,:) + (s*q);
                        
                        pm_C(pm_n_bvs, 1:pm_n_bvs) = 0;
                        pm_C(1:pm_n_bvs, pm_n_bvs) = 0;
                        
                        pm_C(1:pm_n_bvs, 1:pm_n_bvs) = pm_C(1:pm_n_bvs, 1:pm_n_bvs) + r*(s*s.');
                        
                        %update recency scores
                        new_recency_scores = k/kstar;
                        %new_recency_scores = k/kstar .* -1*sqrt(sum((y - m).^2));
                        
                        pm_recency_scores = pm_rfilt.*pm_recency_scores(:) + ...
                            (1-pm_rfilt).*new_recency_scores(:);
                        %pm_recency_scores(pm_n_bvs) = kstar/kstar;
                        pm_recency_scores(pm_n_bvs) = 0.0;
                        
                        
                        %update the kernel matrix
                        
                        pm_KK(1:pm_n_bvs-1, pm_n_bvs) = k;
                        pm_KK(pm_n_bvs, 1:pm_n_bvs-1) = k;
                        pm_KK(pm_n_bvs, pm_n_bvs) = kstar-noiseval;
                        pm_UK = triu(pm_KK);
                        pm_kyz = mean(pm_UK(pm_UK ~= 0));

                    else
                        if pm_debug_mode
                            disp('Info: Sparse update');
                        end
                        %sparse update
                        s = Ck +ehat;
                        pm_alpha(1:pm_n_bvs,:) = pm_alpha(1:pm_n_bvs,:) + s*(q*eta);
                        
                        pm_C(1:pm_n_bvs,1:pm_n_bvs) = pm_C(1:pm_n_bvs,1:pm_n_bvs) + r*eta*(s*s.');
                        
                        
                        new_recency_scores = k./kstar;
                        %new_recency_scores = k/kstar .* -1*sqrt(sum((y - m).^2));
                        pm_recency_scores = pm_rfilt*(pm_recency_scores(:)) + ...
                            (1-pm_rfilt)*new_recency_scores;
                        
                        
                    end
                    
                end
                
                %basis vector deletion
                if (pm_n_bvs > pm_capacity)
                    %scores
                    %scores = (pm_alpha(1:pm_n_bvs,:).^2) ./ repmat((diag(pm_Q(1:pm_n_bvs,1:pm_n_bvs)) + ...
                    %    diag(pm_C(1:pm_n_bvs,1:pm_n_bvs))), 1, y_dim);
                    
                    %if any(~isfinite(scores))
                    scores = (pm_alpha(1:pm_n_bvs,:).^2) ./ repmat((diag(pm_Q(1:pm_n_bvs,1:pm_n_bvs))), 1, y_dim);
                    %end
                    
                    if pm_deletion_criteria == 'n'
                        reduced_scores = pm_rowNorm(scores,2);
                    elseif pm_deletion_criteria == 'm'
                        reduced_scores = max(scores');
                    end
                    
                    finite_scores = scores(isfinite(reduced_scores));
                    reduced_scores = reduced_scores ./ sum(finite_scores);
                    finite_scores = reduced_scores(isfinite(reduced_scores));
                    
                    %compute local recency scores
                    lrs = pm_recency_scores ./ nansum(pm_recency_scores);
                    
                    %normalization
                    normalize_scores = true;
                    if normalize_scores
                        diff_res = max(finite_scores) - min(finite_scores);
                        if diff_res == 0
                            diff_res = 1;
                        end
                        reduced_scores = (reduced_scores - min(finite_scores)) ./ diff_res;
                        lrs = (lrs - min(lrs)) ./ (max(lrs) - min(lrs));
                    end
                    
                    %all = [sort(reduced_scores) sort(lrs)]
                    
                    
                    %                 diff_lrs = max(lrs) - min(lrs);
                    %                 lrs = (lrs - min(lrs)) ./diff_lrs;
                    %                 lrs = (lrs - mean(lrs)) ./ var(lrs);
                    %                 diff_lrs = max(lrs) - min(lrs);
                    %                 lrs = (lrs - min(lrs)) ./diff_lrs;
                    
                    total_score = pm_sw*reduced_scores + ...
                        (1-pm_sw)*(-lrs);
                    
                    
                    [~, min_index] = min(total_score);
                    pm_deleteBasisVector( min_index );
                    
                end
            end
            
            train_x_buffer = [];
            train_y_buffer = [];
        end
        
        %update interval
        pm_t = pm_t + 1;
        
    end



    function [pred_mean, pred_var, log_marg_likl, rebuilt] = pm_train(x_in, y_in)
        %TRAIN Trains the MemOGP2 using input x and output y

        if nargin == 2
            x = x_in;
            y = y_in;
        elseif nargin == 1
            x = pm_memory.getState();
            y = x_in;

        else
            error('Error using Training function');
            return;
        end
        rebuilt = false;

        n_params = length(pm_covf_params);

        %check if we are supposed to optimise hyperparameters
        %if and(pm_optimize_params ,pm_t > pm_capacity + 1)
        if and(pm_n_bvs > 0, pm_t > length(pm_covf_params)*2) %~= 0 %heuristic, only start optimising if we have some elements
            %check if we are supposed to optimise hyperparameters
            %if and(pm_optimize_params ,pm_t > pm_capacity + 1)
            if pm_optimize_params
                %yes, let's check the current "optimization time"

                %compute the gradients
                th = mod(pm_t, length(pm_hyp_cell)) + 1;

                [gradients] = pm_getHypGradients(y, pm_hyp_cell{th});

                if isempty(pm_gradients)
                    pm_gradients = gradients;
                else
                    pm_gradients = pm_gradients + gradients;
                end


                if th == length(pm_hyp_cell) %finished updating all gradients

                        if length(y) > 1
                            gradients = nansum(pm_gradients)';
                        else
                            gradients = pm_gradients;
                            gradients(isnan(pm_gradients)) = 0;
                            gradients = gradients';
                        end


                    %pm_gradients
                    %gradients
                    %update the mean of the hyp_gradients
                    if any(isnan(gradients))
                        warning('Warning! Gradient is NaN! Not including in this update');
                    else
                        pm_optAlg.update(gradients);
                        pm_gradients = [];
                    end
                end

                %perform whole model optimization if necessary
                if pm_opt_interval > 0
                    if mod(pm_t, length(pm_hyp_cell)*pm_opt_interval) == 0
                        pm_optimizeModel();
                        pm_gradients = [];
                        rebuilt = true;
                    end
                end

                
            end
            
                            
            kstar = feval(pm_covf{:},pm_covf_params,x);
            
            %end
            y_dim = size(y,2);
            
            
            k = feval(pm_covf{:},pm_covf_params, pm_basis_vectors(1:pm_n_bvs, :), x);
            m = k'*pm_alpha(1:pm_n_bvs, :);
            
            if any(isnan(m))
                x
                kernel_params = pm_covf_params
                k
                pm_alpha
                error('MemOGP2: Training. Predicted mean is NaN');
            end
            
            Ck = pm_C(1:pm_n_bvs, 1:pm_n_bvs)*k;
            s2 = kstar + (k.'*Ck);
            
            if (s2 < 1e-12)
                if (pm_debug_mode)
                    fprintf(1, 'WARNING: variance too small. Possible numerical error. Try increasing tolerance?: %f\n',s2);
                end
                s2 = minvar;
            end
            
            
            pred_mean = m;
            pred_var = s2;
            pred_var = zeros(1,y_dim) + pred_var;
            
            log_marg_likl = log(normpdf(y, m, sqrt(s2)));
            
           
            
        else
            rebuilt = true;
        end

        
        
        
        %continue on with training
        %if not(pm_optimize_params)

        kstar = feval(pm_covf{:},pm_covf_params,x);

        %end
        y_dim = size(y,2);


        %is this a new pmgp?
        if pm_n_bvs == 0
            pm_alpha = y / (kstar) ;
            pm_ys = y;
            pm_C = -1/(kstar); %+ pm_noise);
            noiseval =kern_noise_func(pm_covf_params);
            pm_Q = 1/(kstar - noiseval);
            pm_KK = kstar - noiseval;
            pm_UK = pm_KK;
            pm_kyz = pm_KK;
            
            pm_basis_vectors(1,:) = x;
            bv_mean = x;
            pm_n_bvs = 1;
            pred_mean = zeros(1, y_dim);
            pred_var = zeros(1, y_dim);
            log_marg_likl = log(normpdf(y, 0, kstar));
            pm_recency_scores = [kstar];
            pm_t = 0;  
            
        else

            %nope, let's do a geometric test

            %if not(pm_optimize_params)
            k = feval(pm_covf{:},pm_covf_params, pm_basis_vectors(1:pm_n_bvs, :), x);
            m = k'*pm_alpha(1:pm_n_bvs, :);

            if any(isnan(m))
                x
                kernel_params = pm_covf_params
                k
                pm_alpha
                error('MemOGP2: Training. Predicted mean is NaN');
            end

            Ck = pm_C(1:pm_n_bvs, 1:pm_n_bvs)*k;
            s2 = kstar + (k.'*Ck);
            
            if (s2 < 1e-12)
                if (pm_debug_mode)
                    fprintf(1, 'WARNING: variance too small. Possible numerical error. Try increasing tolerance?: %f\n',s2);
                end
                s2 = minvar;
            end
            
            
            pred_mean = m;
            pred_var = s2;
            pred_var = zeros(1,y_dim) + pred_var;

            log_marg_likl = log(normpdf(y, m, sqrt(s2)));
            %log_marg_likl = -normlike([m, sqrt(s2)], y);



            if pm_inf_mode == 'r'
                r = -1.0/(s2);
                q = -r*(y - m);
            elseif pm_inf_mode == 'c'

%                     if s2 < noise
%                         s2 = noise;
%                     end

                sx = sqrt(s2);
                pred_Erfz = pm_Erf(m/sx);

                if pred_Erfz > 0.5
                    pred_mean = 1;
                else
                    pred_mean = -1;
                end
                pred_var = pred_Erfz;
                z0 = m/sx;
                z = y.*z0;

                Erfz = pm_Erf(z);
                if abs(Erfz) < 1e-20
                    q = 0.0;
                    r = 0.0;
                else
                    dErfz = pm_regularizer/sqrt(2*pi)*exp(-(z.^2)/2);
                    dErfz2 = pm_regularizer*dErfz.*(-z);

                    if abs(dErfz) < 1e-20
                        q = 0.0;
                        r = 0.0;
                    else
                        q = y./sx .* (dErfz./Erfz);
                        r = (1/s2)*(dErfz2./dErfz - (dErfz./Erfz)^2);
                    end
                end
            else
                error('Error in training. Inference mode must be either (r)egression or (c)lassfication');
            end

            
            
            
            if isnan(r) || isinf(r)
                r
                q
                y
                m
                s2
                error('r is NaN');
            end


            ehat = pm_Q(1:pm_n_bvs,1:pm_n_bvs)*k;
            noiseval = kern_noise_func(pm_covf_params);
            gamma = kstar - dot(k,ehat) - noiseval; %modified because the kernel now includes the noise term
            
            

            if (gamma < 1e-12 )
                if (pm_debug_mode)
                    fprintf(1, 'WARNING: gamma too small. Possible numerical issue. Try increasing tolerance?: %f\n', gamma);
                end
                gamma = 0;
            end
            eta = 1.0./(1.0 + gamma*r);
            
            if (gamma >= pm_epsilon*(kstar-noiseval))
                %full update
                s = [Ck; 1];

                
                %add to bv
 
                pm_n_bvs = pm_n_bvs+1;
                bv_mean = (1/pm_n_bvs)*x + ((pm_n_bvs-1)/pm_n_bvs)*bv_mean;
                
                pm_basis_vectors(pm_n_bvs,:) = x;
                
                pm_ys(pm_n_bvs,:) = y;

                %update Q
                pm_Q(pm_n_bvs, 1:pm_n_bvs) = 0;
                pm_Q(1:pm_n_bvs, pm_n_bvs) = 0;

                ehat(pm_n_bvs) = -1;
                ehat = ehat(:); %make ehat a column vector

                pm_Q(1:pm_n_bvs, 1:pm_n_bvs) = pm_Q(1:pm_n_bvs, 1:pm_n_bvs) + (1/gamma)*(ehat*ehat.');

                pm_alpha(pm_n_bvs,:) = 0;
                pm_alpha(1:pm_n_bvs,:) = pm_alpha(1:pm_n_bvs,:) + (s*q);



                pm_C(pm_n_bvs, 1:pm_n_bvs) = 0;
                pm_C(1:pm_n_bvs, pm_n_bvs) = 0;

                pm_C(1:pm_n_bvs, 1:pm_n_bvs) = pm_C(1:pm_n_bvs, 1:pm_n_bvs) + r*(s*s.');

                %update recency scores
                new_recency_scores = k/kstar;
                %new_recency_scores = k/kstar .* -1*sqrt(sum((y - m).^2));

                pm_recency_scores = pm_rfilt.*pm_recency_scores(:) + ...
                    (1-pm_rfilt).*new_recency_scores(:);
                %pm_recency_scores(pm_n_bvs) = kstar/kstar;
                pm_recency_scores(pm_n_bvs) = 0.0;
                
                %update the upper triangular kernel matrix
                %pm_KK = rhs.pm_KK; %feval(pm_covf{:},pm_covf_params, X, X);

                pm_KK(1:pm_n_bvs-1, pm_n_bvs) = k;
                pm_KK(pm_n_bvs, 1:pm_n_bvs-1) = k;
                pm_KK(pm_n_bvs, pm_n_bvs) = kstar-noiseval;
                pm_UK = triu(pm_KK);
                pm_kyz = mean(pm_UK(pm_UK ~= 0));

            else

                %sparse update
                s = Ck +ehat;
                pm_alpha(1:pm_n_bvs,:) = pm_alpha(1:pm_n_bvs,:) + s*(q*eta);
                pm_C(1:pm_n_bvs,1:pm_n_bvs) = pm_C(1:pm_n_bvs,1:pm_n_bvs) + r*eta*(s*s.');
                new_recency_scores = k./kstar;
                %new_recency_scores = k/kstar .* -1*sqrt(sum((y - m).^2));
                pm_recency_scores = pm_rfilt*(pm_recency_scores(:)) + ...
                    (1-pm_rfilt)*new_recency_scores;


            end


        end

        %basis vector deletion
        if (pm_n_bvs > pm_capacity)
            %scores
            %scores = (pm_alpha(1:pm_n_bvs,:).^2) ./ repmat((diag(pm_Q(1:pm_n_bvs,1:pm_n_bvs)) + ...
            %    diag(pm_C(1:pm_n_bvs,1:pm_n_bvs))), 1, y_dim);

            %if any(~isfinite(scores))
            scores = (pm_alpha(1:pm_n_bvs,:).^2) ./ repmat((diag(pm_Q(1:pm_n_bvs,1:pm_n_bvs))), 1, y_dim);
            %end

            if pm_deletion_criteria == 'n'
                reduced_scores = pm_rowNorm(scores,2);
            elseif pm_deletion_criteria == 'm'
                reduced_scores = max(scores');
            end

            
            finite_scores = scores(isfinite(reduced_scores));
            
            reduced_scores = reduced_scores ./ sum(finite_scores);
            finite_scores = reduced_scores(isfinite(reduced_scores));

            %compute local recency scores
            if nansum(pm_recency_scores) >=0
                lrs = pm_recency_scores ./ nansum(pm_recency_scores);
            else
                lrs = pm_recency_scores;
            end
            %normalization
            normalize_scores = true;
            if normalize_scores
                
                diff_res = max(finite_scores) - min(finite_scores);
                reduced_scores = (reduced_scores - min(finite_scores)) ./ diff_res;
                lrs = (lrs - min(lrs)) ./ (max(lrs) - min(lrs));
            end

            %all = [sort(reduced_scores) sort(lrs)]


            %                 diff_lrs = max(lrs) - min(lrs);
            %                 lrs = (lrs - min(lrs)) ./diff_lrs;
            %                 lrs = (lrs - mean(lrs)) ./ var(lrs);
            %                 diff_lrs = max(lrs) - min(lrs);
            %                 lrs = (lrs - min(lrs)) ./diff_lrs;

            total_score = pm_sw*reduced_scores + ...
                (1-pm_sw)*(-lrs);


            [~, min_index] = min(total_score);
            pm_deleteBasisVector( min_index );

        end
        pm_t = pm_t + 1;

    end


    function [ pmgp ] = pm_deleteBasisVector(min_index  )
        %DELETEBASISVECTOR Deletes a basis vector from the MemOGP2
        %   This is an internal function and usually does not need to
        %   be called explicitly
        
        %correct basis vectors and y's
        pm_basis_vectors(min_index,:) = pm_basis_vectors(pm_n_bvs,:);
        
        bv_mean = (pm_n_bvs)/(pm_n_bvs-1).*(bv_mean) - pm_basis_vectors(min_index,:)./pm_n_bvs;
        
        pm_ys(min_index, :) = pm_ys(pm_n_bvs,:);
        
        %fix alpha
        alphastar = pm_alpha(min_index,:);
        pm_alpha(min_index, :) = pm_alpha(pm_n_bvs, :);
        
        %get cstar and Cstar (and for Q as well
        cstar = pm_C(min_index, min_index);
        Cstar = pm_C(1:(pm_n_bvs-1), min_index);
        
        qstar = pm_Q(min_index, min_index);
        Qstar = pm_Q(1:(pm_n_bvs-1), min_index);
        
        kkstar = pm_KK(min_index, min_index);
        KKstar = pm_KK(1:(pm_n_bvs-1), min_index);
        %if the min index is not the last one, we need to shift things
        %around
        if (min_index ~= pm_n_bvs)
            Cstar(min_index) = pm_C(pm_n_bvs, min_index);
            Crep = pm_C(1:(pm_n_bvs-1), pm_n_bvs);
            Crep(min_index) = pm_C(pm_n_bvs, pm_n_bvs);
            
            pm_C(min_index, 1:(pm_n_bvs-1)) = Crep';
            pm_C(1:(pm_n_bvs-1), min_index) = Crep;
            
            Qstar(min_index) = pm_Q(pm_n_bvs, min_index);
            Qrep = pm_Q(1:(pm_n_bvs-1), pm_n_bvs);
            Qrep(min_index) = pm_Q(pm_n_bvs, pm_n_bvs);
            
            pm_Q(min_index, 1:(pm_n_bvs-1)) = Qrep';
            pm_Q(1:(pm_n_bvs-1), min_index) = Qrep;
            
            
            KKstar(min_index) = pm_KK(pm_n_bvs, min_index);
            KKrep = pm_KK(1:(pm_n_bvs-1), pm_n_bvs);
            KKrep(min_index) = pm_KK(pm_n_bvs, pm_n_bvs);
            
            pm_KK(min_index, 1:(pm_n_bvs-1)) = KKrep';
            pm_KK(1:(pm_n_bvs-1), min_index) = KKrep;
              
        end
        
        qc = (Qstar)/(qstar);
        for i=1:size(pm_alpha,2)
            pm_alpha(1:(pm_n_bvs-1),i) = pm_alpha(1:(pm_n_bvs-1),i) - alphastar(i)*qc;
        end
        
        QQstar = Qstar*Qstar';
        QQstar_qstar = (QQstar)/qstar;
        QCstar = Qstar*Cstar';
        QCstar = QCstar + QCstar';
        
        
        new_bvs = pm_n_bvs - 1;
        
        pm_C(1:new_bvs,1:new_bvs) = pm_C(1:new_bvs,1:new_bvs) + (cstar/qstar)*QQstar_qstar - ...
            ((QCstar))/(qstar);
        pm_Q(1:new_bvs, 1:new_bvs) = pm_Q(1:new_bvs, 1:new_bvs) - QQstar_qstar;
        
        pm_UK = triu(pm_KK(1:new_bvs, 1:new_bvs));
        pm_kyz = mean(pm_UK(pm_UK ~= 0));

        pm_recency_scores(min_index) = pm_recency_scores(end);
        pm_recency_scores = pm_recency_scores(1:end-1);
        
        pm_n_bvs = new_bvs;
        
        
    end


    function [ pred_mean, pred_var ] = pm_predict(noise)
        %PREDICT Performs a prediction.
        %   Call predict on the internal memory to obtain a mean
        %   prediction. If two outputs are specified, the function also
        %   returns the variance of the prediction.
        
        x = pm_memory.getState();
        
        if isempty(pm_basis_vectors)
            pred_mean = 0;
            if (nargout > 1)
                kstar = feval(pm_covf{:},pm_covf_params,x);
                pred_var = kstar;
            end
            
            %if classifying, we need to do more
            if pm_inf_mode == 'c'
                %                 if pred_var <= noise
                %                     pred_var = noise;
                %                 end
                
                z = pred_mean/pred_var;
                Erfz = pm_Erf(z);
                if Erfz > 0.5
                    pred_mean = 1;
                else
                    pred_mean = -1;
                end
                pred_var = Erfz;
            end
            
            return
        end
        
        Kss = feval(pm_covf{:},pm_covf_params,pm_basis_vectors(1:pm_n_bvs,:), x);
        pred_mean = Kss'*pm_alpha(1:pm_n_bvs,:);
        
        if pm_inf_mode == 'r'
            
            %Return the var if the calling function wants it
            if (nargout > 1)
                kstar = feval(pm_covf{:},pm_covf_params,x);
                pred_var =  kstar + Kss'*(pm_C(1:pm_n_bvs,1:pm_n_bvs)*Kss);
                if (pred_var <= 1e-6)
                    %warn_str = sprintf('MemOGP2: predicted variance is %f, which is <=0. Possible numerical failure', pred_var);
                    %warning(warn_str);
                    pred_var = minvar;
                end
                pred_var = zeros (1,length(pred_mean)) + pred_var;
            end
            
        elseif pm_inf_mode == 'c'
            kstar = feval(pm_covf{:},pm_covf_params,x);
            
            pred_var =  kstar + Kss'*(pm_C(1:pm_n_bvs, 1:pm_n_bvs)*Kss);% +pm_noise;
            %                 if (pred_var <= noise)
            %                     %warn_str = sprintf('MemOGP2: predicted variance is %f, which is very small. Possible numerical failure', pred_var);
            %                     %warning(warn_str);
            %                     pred_var = noise;
            %                 end
            
            if pred_var <= 0
                pred_var = 0.01;
            end
            sx = sqrt(pred_var);
            
            
            %compute the classification
            z = pred_mean/sx;
            
            Erfz = pm_Erf(z);
            if Erfz > 0.5
                pred_mean = 1;
            else
                pred_mean = -1;
            end
            pred_var = Erfz;
            
        end
        
    end

    function [kern_params] = pm_getKernParams()
        %PM_GETKERNHYP Gets the kernel hyperparameters
        kern_params = pm_covf_params;
    end

    function pm_setKernParams(kern_params)
        %PM_GETKERNHYP Gets the kernel hyperparameters
        pm_covf_params = kern_params;
    end

    function pm_setHypOpt(optimization_enabled)
        %SETHYPOPT Sets whether to optimize parameters or not.
        pm_optimize_params = optimization_enabled;
    end

    function [opt_params] = pm_getHypOpt()
        %SETHYPOPT Sets whether to optimize parameters or not.
        opt_params = pm_optimize_params;
    end


    %memory based functions
    function [memory_state] = pm_updateMemory(x)
        %UPDATEMEMORY updates the internal memory with x
        memory_state = pm_memory.update(x);
    end


    function [memory_state] = pm_resetMemory()
        %RESETMEMORY updates the internal memory
        memory_state = pm_memory.resetState();
    end


    function erfz = pm_Erf(z)
        erfz = normcdf(z,0,1);
    end




end


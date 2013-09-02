function [ olm ] = OnlineLocalMemOGP( k_thres, delay_line, base_model, ...
    max_n_models, value_filter, prune_threshold, use_kernel_distance, sigma_c, ...
    use_evt, evt_threshold, start_evt_iteration, optimise_params, combined_estimate)
%ONLINELOCALMEMOGP Summary of this function goes here
%   Detailed explanation goes here
ftype = 'OnlineLocalMemOGP';
olm_base_model = base_model;
olm_dl = delay_line;
olm_k_thres = k_thres;
olm_n_models = 1;
olm_iteration = 0;
olm_max_n_models = max_n_models;

olm_model_value = 1;
last_novel_time = 1;
olm_value_filter = value_filter;
use_kern_dist = use_kernel_distance;
olm_prune_threshold = prune_threshold;
olm_use_evt = use_evt;
olm_evt_threshold = evt_threshold;
olm_optimise_params = optimise_params;
olm_start_evt_iteration = start_evt_iteration;
olm_sigma_c = sigma_c;

olm_combined_estimate = combined_estimate;

%olm_prob_assignment = use_probabilistic_assignment;

p_model = 1.0;
models{1} = FMemOGP2f(olm_base_model);
models{1}.reset(olm_optimise_params);

%function declarations
olm.ftype = ftype;
olm.iteration = olm_iteration;
olm.omodels = models;
olm.getPModels = @olm_getPModels;
olm.update = @olm_update;
olm.updateMemory = @olm_update; %for legacy code
olm.resetMemory = @olm_resetMemory;
olm.train = @olm_train;
olm.predict = @olm_predict;
olm.getNumModels = @olm_getNumModels;
olm.getNumBasisVectors = @olm_getNumBasisVectors;
olm.getBasisVectors = @olm_getBasisVectors;
olm.setHypOpt = @olm_setHypOpt;
olm.getKernParams = @olm_getKernParams;

    function [kparams] = olm_getKernParams()
        for c=1:olm_n_models
            kparams{c} = models{c}.getKernParams();
        end
    end

    function olm_setHypOpt(opt_flag)
        for c=1:olm_n_models
            models{c}.setHypOpt(opt_flag);
        end
    end

    function [pm] = olm_getPModels()
        pm = p_model;
    end


    function olm_update(x)
        olm_dl.update(x);
    end

    function olm_resetMemory()
        olm_dl.resetState();
    end

    function [y_pred, y_var, maxscoreidx] = olm_train(y, interval_training)
        
        if nargin == 1
            interval_training = false;
        end
        
        if use_kern_dist
            
            for c=1:olm_n_models
                scores(c) = models{c}.getKernelDistanceScore();
            end
            
            scores = exp(-scores/olm_sigma_c);
            %scores
            
            [maxscoreval, maxscoreidx] = max(scores);
            
        else
            for c=1:olm_n_models
                scores(c) = models{c}.getRelativeDistanceScore();
            end
            
            [maxscoreval, maxscoreidx] = max(scores);
        end
        
        
        %scores
        if olm_iteration == 0
            if interval_training
                [y_pred, y_var, ~, ~] = models{1}.trainInterval(y);
            else
                [y_pred, y_var, ~, ~] = models{1}.train(y);
            end
        else
            to_train_idx = maxscoreidx;
            create_new_model = false;
            
            %first check if we are too far away
            %get prediction from the best model
            
            %we have to get predictions from all "competing" models
            %full approach
            
            [y_pred, y_var] = models{maxscoreidx}.predict();
            
            if olm_iteration > olm_start_evt_iteration
                if olm_n_models < olm_max_n_models
                    if olm_use_evt
                        
                        
                        %maxpscoreval = sortvals(1);
                        if maxscoreval < olm_k_thres
                            %just create
                            %warning('KMEANS CREATION');
                            create_new_model = true;
                            
                        else
                            
                            % full approach
                            %get all predictions
                            for i=1:olm_n_models
                                [ty_pred(i,:), ty_var(i,:)] = models{i}.predict();
                            end
                            
                            
                            %get the probability over classes given output
                            %uniform prior
                            p_model(i) = 1.0/olm_n_models;
                            
                            for i=1:olm_n_models
                                
                                
                                %p_model(i) = 0.99*p_model(i) + 0.01/olm_n_models;

                                likelihood =  prod(normpdf(y,...
                                    ty_pred(i,:), ...
                                    ty_var(i,:)));
                                
                                if isfinite(likelihood)
                                    p_model(i) = likelihood*scores(i)*p_model(i);
                                else
                                    p_model(i) = 0;
                                end
                                
                            end
                            
                            if sum(p_model) == 0
                                %warning('is nan');
                                p_model = (1/olm_n_models) + zeros(1,olm_n_models);
                            else
                                p_model = p_model ./ sum(p_model);
                            end

                            [~, sortidxs] = sort(p_model, 'descend');
                            y_pred = ty_pred(sortidxs(1), :);
                            y_var = ty_var(sortidxs(1), :);
                            
                            maxscoreidx = sortidxs(1);
                            
                            
                            sigma = diag(y_var);
                            invsigma = diag(1./y_var);
                            m = olm_iteration; % - last_novel_time;
                            
                            n=size(y_pred,2);
                            
                            SqrtDet = sqrt(det(sigma));
                            C_n = (2*pi)^(n/2) .* SqrtDet;
                            
                            diff = (y - y_pred);
                            fy = (1/C_n)*exp(- 0.5*diff*invsigma*diff');
                            [c_m, alpha_m] = EVT_GaussianEVD_FindParams(sigma, m);
                            
                            Ge = EVT_GaussianEVD_Ge(fy, c_m, alpha_m);
                            
                            
                            novelty_score = 1 - Ge;
                            
                            if isfinite(novelty_score)
                                if novelty_score > olm_evt_threshold
                                    %warning('EVT CREATION');
                                    create_new_model = true;
                                end
                            end
                        end
                        
                    else
                        %older method, based on thresholds
                        %check if we need to create a new model
                        
                        
                        if use_kern_dist
                            if maxscoreval < olm_k_thres
                                create_new_model = true;
                            end
                        else
                            if maxscoreval < olm_k_thres
                                create_new_model = true;
                            end
                        end
                        
                    end
                end
            end
            
            
            
            
            %create new model if needed
            if create_new_model
                %create new model
                olm_n_models = olm_n_models + 1;
                
                models{olm_n_models} = FMemOGP2f(models{maxscoreidx});
                models{olm_n_models}.reset(olm_optimise_params);
                kern_params = models{olm_n_models}.getKernParams();
                kern_params(end) = 0.5*log(0.1);
                models{olm_n_models}.setKernParams(kern_params);
                
                %models{olm_n_models} = FMemOGP2e(olm_base_model);
                
                p_model = 0.01*p_model;
                p_model(end+1) = 0.99;
                p_model = p_model ./ sum(p_model);
                
                if not(isfinite(p_model))
                    error('p_model is no longer finite');
                end
                
                %p_model
                to_train_idx = olm_n_models;
                maxscoreidx = olm_n_models;
                last_novel_time = olm_iteration;
            end
            
            %train appropriately
            if interval_training
                models{to_train_idx}.trainInterval(y);
            else
                models{to_train_idx}.train(y);
            end
            
        end
        
        %update model scores
        olm_model_value(maxscoreidx) = 1.0;
        olm_model_value = (olm_value_filter)*olm_model_value;
        
        if olm_prune_threshold > 0
            %check if any score is less than the threshold
            to_prune = find(olm_model_value < olm_prune_threshold);
            
            %go through all the models to prune and add their basis vectors
            %to the other models
            if not(isempty(to_prune))
                models(to_prune) = [];
                olm_model_value(to_prune) = [];
                p_model(to_prune) = [];
            end
            olm_n_models = length(models);
        end
        
        olm_iteration = olm_iteration+1;
    end

    function [y_pred, y_var, maxscoreidx] = olm_predict()
        
        %find "closest" model
        if olm_combined_estimate
            
            for c=1:olm_n_models
                scores(c) = models{c}.getKernelDistanceScore();
            end
            %scores = exp(-scores/olm_sigma_c);
            [~, maxscoreidx] = min(scores);
            
            for c=1:olm_n_models
                [ty_pred(c,:), ty_var(c,:)] = models{c}.predict();
                noise(c) = models{c}.getNoise();
                ty_var(c,:) = ty_var(c,:) - noise(c);
            end
            
            %assumes 1 dimension
            kstar = models{maxscoreidx}.getKStar();

            cm = (-(olm_n_models - 1)*(1./kstar) + sum(1./ty_var(:,1)));
               
            y_pred = (1/cm)*sum( ( 1./ty_var(:,1) ) .* ty_pred  );
            y_var = zeros(size(ty_var(1,:))) + 1./cm + noise(maxscoreidx);
             
            
        else
            if use_kern_dist
                for c=1:olm_n_models
                    scores(c) = models{c}.getKernelDistanceScore();
                end
                scores = exp(-scores/olm_sigma_c);
                [maxscoreval, maxscoreidx] = max(scores);
            else
                for c=1:olm_n_models
                    scores(c) = models{c}.getRelativeDistanceScore();
                end
                
                [maxscoreval, maxscoreidx] = max(scores);
            end
            [y_pred, y_var] = models{maxscoreidx}.predict();
        end
        
        %         [y_pred, y_var] = models{maxscoreidx}.predict();
        %         for i=1:olm_n_models
        %             [ty_pred, ty_var] = models{i}.predict();
        %         end
        %         [y_pred, ~] = ProductOfNormals(ty_pred, ty_var);
        
        
    end

    function [num_models] = olm_getNumModels()
        num_models = olm_n_models;
    end

    function [nbvs] = olm_getNumBasisVectors()
        nbvs = zeros(olm_n_models, 1);
        for c=1:olm_n_models
            nbvs(c) = models{c}.getNumBasisVectors();
        end
    end

    function [bvs] = olm_getBasisVectors()
        bvs = cell(olm_n_models, 1);
        for c=1:olm_n_models
            bvs{c} = models{c}.getBasisVectors();
        end
    end

end


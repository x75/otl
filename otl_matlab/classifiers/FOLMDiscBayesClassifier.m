function gbc = FOLMDiscBayesClassifier( num_classes, k_thres, delay_line, base_model, ...
    max_n_models, value_filter, prune_threshold, use_kernel_distance,...
    use_evt, evt_threshold,optimise_params)
%GENBAYESCLASSIFIER Generative Bayesian Classifier using compatible
%algorithms as a modelling building block

%internal variables
num_models = num_classes;
models = {}; %the models
p_models = [];
likelihoods = [];
p_models_not_normalised = [];
curr_itr = 0;

TRAIN_SEC = 3; %train closest competing model
TRAIN_DIS = 1; %train competing model when wrong classificaiton occurs
TRAIN_ALL = 2;
TRAIN_POS = 0;

% create models
for i=1:num_models
    models{i} = OnlineLocalMemOGP(k_thres, delay_line, base_model, max_n_models, ...
        value_filter, prune_threshold, 1.0, use_kernel_distance, use_evt, evt_threshold, 100,...
    optimise_params, false);

% k_thres, delay_line, base_model, ...
 %   max_n_models, value_filter, prune_threshold, use_kernel_distance, sigma_c, ...
  %  use_evt, evt_threshold, start_evt_iteration, optimise_params, combined_estimate)

end

resetProb();
resetMemory();

gbc.resetProb = @resetProb;
gbc.resetMemory = @resetMemory;
gbc.reset = @reset;
gbc.update = @update;
gbc.train = @train;
gbc.trainSeq = @trainSeq;
gbc.coTrainSeq = @coTrainSeq;
gbc.omodels = models;
gbc.TRAIN_DIS = TRAIN_DIS;
gbc.TRAIN_ALL = TRAIN_ALL;
gbc.TRAIN_POS = TRAIN_POS;
gbc.TRAIN_SEC = TRAIN_SEC;

    function resetProb()
        p_models = ones(1,num_models)./num_models;
        p_models_not_normalised = ones(1,num_models)./num_models;
    end

    function resetMemory()
        delay_line.resetState();
    end

    function reset()
        resetProb();
        resetMemory();
    end


    function [best_class, p_class, y_pred, y_var] = update( inputs, outputs )
        %update summary of this function goes here
        %   detailed explanation goes here
        
        curr_itr = curr_itr+1;
        %update the models
        
        delay_line.update(inputs);

        
        %make the predictions
        likelihoods = zeros(1,num_models);
        output_dim = size(outputs,2);
        %y_pred = zeros(num_models, output_dim);
        %y_var = zeros(num_models, output_dim);
        for i=1:num_models
            [y_pred, y_var(i,:)] = models{i}.predict();
            if y_pred == 0
                y_pred(i,:) =  -1;
                y_var(i,:) = 0.5;
            end
            
            likelihoods(i) = y_var(i);
            
        end
        
%         p_models
        %likelihoods
        %p_models_not_normalised = p_models_not_normalised.* likelihoods;
        %p_models = p_models_not_normalised ./ sum(p_models_not_normalised);
        p_models = likelihoods ./sum(likelihoods);
        %p_models = likelihoods;
        [~, best_class] = max(p_models);
        
        p_class = p_models;
        
    end

    function train(inputs, outputs, class_label)
        
        models{class_label}.updateMemory(inputs);
        models{class_label}.train(outputs);
        
    end

    function coTrainSeq(inputs, outputs, class_label)
        error('Deprecated');
    end

    function trainSeq(input_seq, output_seq, class_label, train_mode)
        
        if nargin == 3
            train_mode = TRAIN_POS;
        end
        
        
        resetMemory();
        
        for i=1:size(input_seq,1)
            delay_line.update(input_seq(i,:));
            
            %if train_mode 
                for n=1:num_models
                    [y_pred, y_var] = models{n}.predict();
                    if y_pred == 0
                        y_pred =  -1;
                        y_var = 0.5;
                    end
                    
                    pmodel(n) = y_var;
                end
                
                
                %find wrongly selected model
                [~, sorted_idxs] = sort(pmodel, 'descend');
                best_class = sorted_idxs(1);
                second_best = sorted_idxs(2);
                %[~, best_class] = max(pmodel);
            %end
            
            
            
            for n=1:num_models
                if n == class_label
                    models{class_label}.train(1, true);
                elseif train_mode == TRAIN_ALL
                    %all training so, we update all the models
                    models{n}.train(-1, true);
                elseif train_mode == TRAIN_DIS
                    %we have to find the "wrongly selected model"
                    if best_class ~= class_label
                        if n == best_class
                            models{n}.train(-1, true);
                        end    
                    end
                elseif train_mode == TRAIN_SEC
                    if n == second_best
                    	models{n}.train(-1, true);
                    end  
                end
            end
        end
        
%         if nargin < 5
%             optimize_generative_models = true;
%         end
%         
%         if optimize_generative_models
%             models{class_label}.optimizeModel();
%         end
        
        resetMemory();
    end


end


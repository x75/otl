function gbc = FOLMGenBayesClassifier( num_classes, k_thres, delay_line, base_model, ...
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
gbc.omodels = models;

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
        likelihoods = ones(1,num_models);
        output_dim = size(outputs,2);
        y_pred = zeros(num_models, output_dim);
        y_var = zeros(num_models, output_dim);
        for i=1:num_models
            [y_pred(i,:), y_var(i,:)] = models{i}.predict();
            if y_pred == 0
                y_pred(i,:) = zeros(size(outputs));
                y_var(i,:) = zeros(size(outputs)) + 10;
            end
            
            likelihoods(i) = prod( normpdf(outputs, ...
                y_pred(i,:), sqrt(y_var(i,:))));
            
        end
        
%         p_models
%         likelihoods
        p_models_not_normalised = p_models_not_normalised.* likelihoods;
        p_models = p_models_not_normalised ./ sum(p_models_not_normalised);
        
        [~, best_class] = max(p_models);
        
        p_class = p_models;
        
    end

    function train(inputs, outputs, class_label)
        
        models{class_label}.updateMemory(inputs);
        models{class_label}.train(outputs);
        
    end


    function trainSeq(input_seq, output_seq, class_label)
        
        resetMemory();
        
        for i=1:size(input_seq,1)
            delay_line.update(input_seq(i,:));
            models{class_label}.train(output_seq(i,:), true);
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


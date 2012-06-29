classdef OESGP < handle
    %OESGP oesgp algorithm by Soh
    %
    
    properties
        input_dim;  %input dimension
        output_dim; %output dimension
        reservoir_size;
        input_weight;
        output_feedback_weight;
        activation_function;
        leak_rate;
        connectivity, spectral_radius;
        use_inputs_in_state;
        kernel_parameters;
        noise;
        epsilon;
        capacity;
        random_seed;
        
        
        sogp;       %sogp object
        num_processed; %number of elements processed
    end
    
    methods
        function oesgp = OESGP( input_dim, output_dim, reservoir_size, ...
                    input_weight, output_feedback_weight,...
                    activation_function,...
                    leak_rate,...
                    connectivity, spectral_radius,...
                    use_inputs_in_state,...
                    kernel_parameters,...
                    noise, epsilon, capacity, random_seed)
            %OESGP initialiss the OESGP object            
            oesgp.input_dim = input_dim;
            oesgp.output_dim = output_dim;
            
            act_func = 0;
            if strcmp(activation_function, 'TANH')
                act_func = 0;
            elseif strcmp(activation_function, 'LINEAR')
                act_func = 1;
            else 
                disp 'No such activation function. Defaulting to TANH'
                act_func = 0; %default
            end
            
            oesgp.sogp = createOESGP(input_dim, output_dim, ...
                    reservoir_size,...
                    input_weight, output_feedback_weight,...
                    act_func,...
                    leak_rate,...
                    connectivity, spectral_radius,...
                    use_inputs_in_state,...
                    kernel_parameters,...
                    noise, epsilon, capacity, random_seed);

            oesgp.num_processed = 0;
            return;
        end
        
        function delete(oesgp)
           destroyOESGP(oesgp.sogp); 
        end
        
        function train( oesgp, y )
            %train trains the model given the current memory state and 
            %a given observation
            %   y: a row vector of output_dim length

            trainOESGP(oesgp.sogp, y);
            oesgp.num_processed = oesgp.num_processed + 1;
            
        end
        
        function [ pred_mean, pred_var ] = predict( oesgp )
            %predict_psogp summary of this function goes here
            %   detailed explanation goes here
            
            [pred_mean, pred_var] = predictOESGP(oesgp.sogp);
            
        end
        
        function update( oesgp, x )
            %update updates the memory of the oesgp method with the
            %input
            %   x: a row vector of input_dim length
            updateOESGP(oesgp.sogp, x);
            
        end
       
        function save( oesgp, filename)
            %resetState resets the memory state of the OESGP model.
            %NOTE: this does not reset the entire model, just the memory.
            saveOESGP(oesgp.sogp, filename);
        end

        function load( oesgp, filename)
            %resetState resets the memory state of the OESGP model.
            %NOTE: this does not reset the entire model, just the memory.
            loadOESGP(oesgp.sogp, filename);
        end

 
        function resetState( oesgp)
            %resetState resets the memory state of the OESGP model.
            %NOTE: this does not reset the entire model, just the memory.
            resetOESGP(oesgp.sogp);
        end
    end
    
end


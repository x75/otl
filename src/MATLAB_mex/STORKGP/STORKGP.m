classdef STORKGP < handle
    %STORKGP stork-gp algorithm by Soh
    %
    
    properties
        input_dim;  %input dimension
        output_dim; %output dimension
        tau;        %memory length
        sogp;       %sogp object
        num_processed; %number of elements processed
    end
    
    methods
        function storkgp = STORKGP( input_dim, output_dim, ...
                tau, ...
                kern_params, ...
                noise, epsilon, capacity)
            %STORKGP initialiss the STORKGP object
            %   input_dim input dimension
            %   output_dim output dimension
            %   tau memory length
            %   kern_params kernel parametres [l, rho, alpha]
            %       l = characteristic length
            %       rho = spectral radius (typically 0.90)
            %       alpha = scale (just leave this to 1.0)
            %   noise estimated noise level
            %   epsilon approximation level (typically small, e.g. 1e-5)
            %   capacity size of the model (application dependent, try 100)
            
            
            storkgp.input_dim = input_dim;
            storkgp.output_dim = output_dim;
            storkgp.tau = tau;
            
            storkgp.sogp = createSTORKGP( input_dim, output_dim, tau, ...
                kern_params, noise, epsilon, capacity);

            storkgp.num_processed = 0;
            return;
        end
        
        function delete(storkgp)
           destroySTORKGP(storkgp.sogp); 
        end
        
        function train( storkgp, y )
            %train trains the model given the current memory state and 
            %a given observation
            %   y: a row vector of output_dim length

            trainSTORKGP(storkgp.sogp, y);
            storkgp.num_processed = storkgp.num_processed + 1;
            
        end
        
        function [ pred_mean, pred_var ] = predict( storkgp )
            %predict_psogp summary of this function goes here
            %   detailed explanation goes here
            
            [pred_mean, pred_var] = predictSTORKGP(storkgp.sogp);
            
        end
        
        function update( storkgp, x )
            %update updates the memory of the storkgp method with the
            %input
            %   x: a row vector of input_dim length
            updateSTORKGP(storkgp.sogp, x);
            
        end

        function save( storkgp, filename)
            %resetState resets the memory state of the OESGP model.
            %NOTE: this does not reset the entire model, just the memory.
            saveSTORKGP(storkgp.sogp, filename);
        end

        function load( storkgp, filename)
            %resetState resets the memory state of the OESGP model.
            %NOTE: this does not reset the entire model, just the memory.
            loadSTORKGP(storkgp.sogp, filename);
        end

        
        function resetState( storkgp)
            %resetState resets the memory state of the STORKGP model.
            %NOTE: this does not reset the entire model, just the memory.
            resetSTORKGP(storkgp.sogp);
        end
    end
    
end


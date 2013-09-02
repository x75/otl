function dl = FMultiCyclicReservoir(input_dim, ...
    res_size, ...
    input_weight, res_weight)

    if nargin == 1
        rhs = input_dim;
        if isstruct(rhs)
            if strcmp(rhs.ftype, 'FMultiCyclicReservoir')
                disp('Copying FMultiCyclicReservoir.');
                setInternals(rhs.getInternals());
            else
                error('Non-matching FMultiCyclicReservoir type. Cannot copy!');
            end

        else
            help FMultiCyclicReservoir;
        end
    else
        ftype = 'FMultiCyclicReservoir';
        
        %create the reservoirs
        cyclic_res = cell(input_dim);
        for i=1:input_dim
            cyclic_res{i} = FCyclicReservoir(1, res_size, ...
                input_weight, res_weight);
        end
        
        %internal states
        int_state = normrnd(0,1e-8,1,res_size*input_dim);
        ext_int_state = [int_state 1]; %add the bias term
    end
    
    dl.ftype = ftype;
    dl.getStateSize = @getStateSize;
    dl.update = @update;
    dl.getState = @getState;
    dl.resetState = @resetState;
    dl.setInternals = @setInternals;
    dl.getInternals = @getInternals;

    function [ost] = getInternals()
        ost.ftype = ftype;
        ost.int_state = int_state;
        ost.input_dim = input_dim;

        ost.cyclic_res = cyclic_res;
        
        ost.input_weight = input_weight;
        ost.ext_int_state = ext_int_state;
    end

    function setInternals(rhs)
        ftype = rhs.ftype;
        int_state = rhs.int_state;
        input_dim = rhs.input_dim;
        
        input_weight = rhs.input_weight;
        ext_int_state = rhs.ext_int_state;  
        
        cyclic_res = cell(input_dim);
        for k=1:input_dim
            cyclic_res{k} = FCyclicReservoir(rhs.cyclic_res{k});
        end
          
    end
    
    
    function [memory_size] = getStateSize()
        %GETSTATESIZE Returns the delay line size (number of elements
        %in the delay line
        memory_size = res_size*input_dim + 1;
    end
        
    function [curr_state] = update(x)
        %UPDATE Updates the internal state with x
        int_state = zeros(res_size, input_dim);
        for k=1:input_dim
            cyclic_res{k}.update(x(k));
            int_state(:,k) = cyclic_res{k}.getInternalState()';
        end
        
        int_state = reshape(int_state, 1, numel(int_state));
        ext_int_state = [int_state 1];

        if nargout == 1
            curr_state = getState();
        end
    end

    function [curr_state] = getState()
        %GETSTATE returns the current state as a row vector
        curr_state = ext_int_state;
    end
        
    function [curr_state] = resetState(init_val)
        %RESETSTATE Resets the state to init_val. If the init_val
        %argument is not provided, it resets it to all zeros.
        if nargin == 0
            init_val = 0;
        end

        %giving the initial value some small noise prevents numerical errors" with
        %all zeros
        int_state = normrnd(0,1e-8,1,res_size*input_dim) ...
            + init_val;
        ext_int_state = [int_state 1];
        if nargout == 1
            curr_state = getState();
        end
    end
        
end


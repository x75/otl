function dl = FCyclicReservoir(input_dim, ...
    res_size, ...
    input_weight, res_weight, leak_rate, act_func)

    if nargin == 1
        rhs = input_dim;
        if isstruct(rhs)
            if strcmp(rhs.ftype, 'FCyclicReservoir')
                disp('Copying FCyclicReservoir.');
                setInternals(rhs.getInternals());
            else
                error('Non-matching FCyclicReservoir type. Cannot copy!');
            end

        else
            help FCyclicReservoir;
        end
    else
        ftype = 'FCyclicReservoir';
        
        %create the reservoir
        W = zeros(res_size);
        for i=1:res_size-1
            W(i+1, i) = res_weight;
        end
        W(1,res_size) = res_weight;
        
        %input matrix
        V = zeros(res_size, input_dim) + input_weight;
        decexp = char(vpa(pi, res_size*input_dim + 2));
        k=3;
        for i=1:res_size
            for j=1:input_dim
                if str2num(decexp(k)) <= 4
                    V(i,j) = -V(i,j);
                end
                k=k+1;
            end
        end
        
        %internal states
        int_state = normrnd(0,1e-8,1,res_size);
        ext_int_state = [int_state 1]; %add the bias term
        
        if nargin < 5
            leak_rate = 1.0;
        end
            
        if nargin < 6
            %use_tanh by default
            act_func = @tanh;
        end
        
    end
    
    dl.ftype = ftype;
    dl.getStateSize = @getStateSize;
    dl.update = @update;
    dl.getState = @getState;
    dl.resetState = @resetState;
    dl.setInternals = @setInternals;
    dl.getInternals = @getInternals;
    dl.getInternalState = @getInternalState;
    dl.getInputDim = @getInputDim;
    
    function [ind] = getInputDim()
        ind = input_dim;
    end
    

    function [ost] = getInternals()
        ost.ftype = ftype;
        ost.int_state = int_state;
        ost.input_dim = input_dim;
        ost.W = W;
        ost.res_size = res_size;
        ost.res_weight = res_weight;

        ost.V = V;
        ost.input_weight = input_weight;
        ost.ext_int_state = ext_int_state;
        ost.leak_rate = leak_rate;
        ost.act_func = act_func;
    end

    function setInternals(rhs)
        ftype = rhs.ftype;
        int_state = rhs.int_state;
        input_dim = rhs.input_dim;

        W = rhs.W;
        res_size = rhs.res_size;
        res_weight = rhs.res_weight;

        V = V;
        input_weight = rhs.input_weight;
        ext_int_state = rhs.ext_int_state;     
        leak_rate = rhs.leak_rate;
        act_func = rhs.act_func;
    end
    
    
    function [memory_size] = getStateSize()
        %GETSTATESIZE Returns the delay line size (number of elements
        %in the delay line
        memory_size = res_size + 1;
    end
        
    function [curr_state] = update(x)
        %UPDATE Updates the internal state with x

        new_state = (act_func(V*(x') + W*(int_state')))';
        int_state = (leak_rate)*int_state + (1-leak_rate)*new_state;
        ext_int_state = [int_state 1];

        if nargout == 1
            curr_state = getState();
        end
    end

    function [curr_state] = getState()
        %GETSTATE returns the current state as a row vector
        curr_state = ext_int_state;
    end

    function [curr_state] = getInternalState()
        %GETSTATE returns the current state as a row vector
        curr_state = int_state;
    end

        
    function [curr_state] = resetState(init_val)
        %RESETSTATE Resets the state to init_val. If the init_val
        %argument is not provided, it resets it to all zeros.
        if nargin == 0
            init_val = 0;
        end

        %giving the initial value some small noise prevents numerical errors" with
        %all zeros
        int_state = normrnd(0,1e-8,1,res_size) ...
            + init_val;
        ext_int_state = [int_state 1];
        if nargout == 1
            curr_state = getState();
        end
    end
        
end


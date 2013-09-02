function dl = FDelayLine(input_dim, memory_length, alpha)

    if nargin == 1
        rhs = input_dim;
        if isstruct(rhs)
            if strcmp(rhs.ftype, 'FDelayLine')
                disp('Copying FDelayLine.');
                setInternals(rhs.getInternals());
            else
                error('Non-matching FDelayLine type. Cannot copy!');
            end

        else
            help FDelayLine;
        end
    else
        ftype = 'FDelayLine';
        int_state = normrnd(0,1e-8,1,input_dim*memory_length);
        
        if nargin < 3
           alpha = 1.0; 
        end
        
        prev_state =0.0;
        
    end
    
    dl.ftype = ftype;
    dl.getStateSize = @getStateSize;
    dl.update = @update;
    dl.getState = @getState;
    dl.resetState = @resetState;
    dl.setInternals = @setInternals;
    dl.getInternals = @getInternals;
    dl.getInputDim = @getInputDim;

    
    function [ind] = getInputDim()
        ind = input_dim;
    end

    function [ost] = getInternals()
        ost.ftype = ftype;
        ost.int_state = int_state;
        ost.input_dim = input_dim;
        ost.memory_length = memory_length;
    end

    function setInternals(rhs)
        ftype = rhs.ftype;
        int_state = rhs.int_state;
        input_dim = rhs.input_dim;
        memory_length = rhs.memory_length;       
    end
    
    
    function [memory_size] = getStateSize()
        %GETSTATESIZE Returns the delay line size (number of elements
        %in the delay line
        memory_size = size(int_state,2);
    end
        
    function [curr_state] = update(x)
        %UPDATE Updates the internal state with x
        x = (1-alpha)*prev_state + alpha*x;
        int_state = [int_state(input_dim+1:end) x(:)'];

        if nargout == 1
            curr_state = getState();
        end
    end

    function [curr_state] = getState()
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
        int_state = normrnd(0,1e-8,1,input_dim*memory_length) ...
            + init_val;

        if nargout == 1
            curr_state = getState();
        end
    end
        
end


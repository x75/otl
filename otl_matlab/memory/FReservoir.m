function dl = FReservoir(input_dim, ...
    res_size, ...
    input_weight, res_weight, spectral_radius, ...
    connectivity, leak_rate, act_func, use_bias)

    if nargin == 1
        rhs = input_dim;
        if isstruct(rhs)
            if strcmp(rhs.ftype, 'FReservoir')
                disp('Copying FReservoir.');
                setInternals(rhs.getInternals());
            else
                error('Non-matching FReservoir type. Cannot copy!');
            end

        else
            help FReservoir;
        end
    else
        ftype = 'FReservoir';
        
        %create the reservoir
        %W = normrnd(res_size);
        
        %reduce using the spectral radius
        W = rand(res_size,res_size);
        W(W < connectivity) = 0;
        W(W >= connectivity) = 1;
        
        W = randn(res_size).*W;

        % enforce desired spectral radius
        [~,d] = eig(W);
        d = diag(abs(d));
        d = max(d,[],1);
        W = W*spectral_radius/d;

        
        %input matrix
        V = zeros(res_size, input_dim) + input_weight;
        %V = normrnd(0,1, res_size, input_dim);
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
        ost.connectivity = connectivity;
        ost.spectral_radius = spectral_radius;
        ost.use_bias = use_bias;
    end

    function setInternals(rhs)
        ftype = rhs.ftype;
        int_state = rhs.int_state;
        input_dim = rhs.input_dim;

        W = rhs.W;
        res_size = rhs.res_size;
        res_weight = rhs.res_weight;
        connectivity = rhs.connectivity;

        V = V;
        input_weight = rhs.input_weight;
        ext_int_state = rhs.ext_int_state;     
        leak_rate = rhs.leak_rate;
        act_func = rhs.act_func;
        spectral_radius = rhs.spectral_radius;
        use_bias = rhs.use_bias;
    end
    
    
    function [memory_size] = getStateSize()
        %GETSTATESIZE Returns the delay line size (number of elements
        %in the delay line
        if use_bias
            memory_size = res_size + 1;
        else
            memory_size = res_size;
        end
    end
        
    function [curr_state] = update(x)
        %UPDATE Updates the internal state with x

        new_state = (act_func(V*(x') + W*(int_state')))';
        int_state = (leak_rate)*int_state + (1-leak_rate)*new_state;
        if use_bias
            ext_int_state = [int_state 1];
        else
            ext_int_state = int_state;
        end

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
        if use_bias
            ext_int_state = [int_state 1];
        else
            ext_int_state = int_state;
        end
        if nargout == 1
            curr_state = getState();
        end
    end
        
end


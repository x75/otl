function ohgd =  FOHGD(init_theta,  lamb0, mult, ...
                gradient_bounds, min_or_max, plotfunc, auto_plot)
    %OHGD Online Hessian-based Gradient Descent
    %   

    if nargin == 1
        rhs = init_theta;
        if isstruct(rhs)
            if strcmp(rhs.ftype, 'FOHGD')
                %disp('Copying FOHGD!');
                setInternals(rhs.getInternals());
            else
                error('Non-matching FOHGD type. Cannot copy!');
            end
            
        else
            help OHGD;
        end
    else
        ftype = 'FOHGD';
        
        multiplier = mult;
        lambda0 = lamb0;
        mean_theta = init_theta;
        mean_grad = [];
        inv_H = [];
        mean_inv_H = [];
        auto_gradient_bound = true;
        average_invH_across_itr = false;

        ohgd_t = 1;
        itr = 2;
        theta_t = init_theta;
        limit_per = 0.8; %default
        gradient_bound_array = [];
        setGradientBounds('AUTO');
        lower_bounds = zeros(size(init_theta))-Inf;
        upper_bounds = zeros(size(init_theta))+Inf;
        
        
        if nargin >= 4
            setGradientBounds(gradient_bounds);
        end

        minimize = true;
        if nargin >= 5
            if strcmp(min_or_max, 'min')
                minimize = true;
            else
                minimize = false;
            end
        end    

        if nargin <= 5
            plotfunc = @defaultPlotFunc;
            %plotfunc = NaN;
        end

        if nargin <= 6
            auto_plot = false;
        end

        plotFigHandle = NaN;
    end
    
    ohgd.ftype = ftype;
    
    %declare functions
    ohgd.setIterationCount = @setIterationCount;
    ohgd.setBounds = @setBounds;
    ohgd.setLowerBounds = @setLowerBounds;
    ohgd.setUpperBounds = @setUpperBounds;
    ohgd.setGradientBounds = @setGradientBounds;
    ohgd.resetHessian = @resetHessian;
    ohgd.resetMeanGrad = @resetMeanGrad;
    ohgd.updateParams = @updateParams;
    ohgd.update   = @update;
    ohgd.setIterationCount = @setIterationCount;
    ohgd.plotParams = @plotParameters;
    ohgd.getInternals = @getInternals;
    ohgd.setInternals = @setInternals;
    ohgd.resetOpt = @resetOpt;
    ohgd.getParams = @getParams;
    
    function [ost] = getInternals()
        ost.ftype = ftype;
        ost.multiplier = multiplier;
        ost.lambda0 = lambda0;
        ost.mean_theta = mean_theta;
        ost.mean_grad = mean_grad;
        ost.inv_H = inv_H;
        ost.mean_inv_H = mean_inv_H;
        ost.auto_gradient_bound = auto_gradient_bound;
        ost.average_invH_across_itr = average_invH_across_itr;
        ost.ohgd_t = ohgd_t;
        ost.itr = itr;
        ost.theta_t = theta_t;
        ost.limit_per = limit_per; 
        ost.gradient_bound_array = gradient_bound_array;
        ost.lower_bounds = lower_bounds;
        ost.upper_bounds = upper_bounds;
        ost.minimize = minimize;
        ost.plotfunc = plotfunc;
        ost.auto_plot = auto_plot;
        ost.plotFigHandle = plotFigHandle;
    end

    function setInternals(rhs)
        ftype = rhs.ftype;
        multiplier = rhs.multiplier;
        lambda0 = rhs.lambda0;
        mean_theta = rhs.mean_theta;
        mean_grad = rhs.mean_grad;
        inv_H = rhs.inv_H;
        mean_inv_H = rhs.mean_inv_H;
        auto_gradient_bound = rhs.auto_gradient_bound;
        average_invH_across_itr = rhs.average_invH_across_itr;
        ohgd_t = rhs.ohgd_t;
        itr = rhs.itr;
        theta_t = rhs.theta_t;
        limit_per = rhs.limit_per; 
        gradient_bound_array = rhs.gradient_bound_array;
        lower_bounds = rhs.lower_bounds;
        upper_bounds = rhs.upper_bounds;
        minimize = rhs.minimize;
        plotfunc = rhs.plotfunc;
        auto_plot = rhs.auto_plot;
        plotFigHandle = NaN; %we never copy the figure handles
    end

    function resetOpt()
        mean_grad = [];
        inv_H = [];
        mean_inv_H = [];   
        ohgd_t = 1;
    end

    
    %define functions
    function setIterationCount(in_itr)
        %SETITERATIONCOUNT Sets the iteration counter (can be used to
        %reset the iteration count to 1 if you specify no parameters.
        %This is primarily used to restart optimisation if needed.
        if nargin < 2
            in_itr = 1;
        end

        if in_itr <= 0;
            in_itr = 1;
        end
        itr = in_itr;
    end

    function plotParameters()
        if isnan(plotFigHandle) 
            plotFigHandle = figure()
            
        end
        plotfunc(plotFigHandle, theta_t);
    end


    function defaultPlotFunc(figh, params)
        figure(figh);
        bar(figh, params);
        drawnow();
    end

    function [mg, iH] = update(gradient)
        %UPDATE Updates stored mean gradient and inverse Hessian

        %check gradient
        if any(isnan(gradient))
            %warning('OHGD: Gradient is NaN. Applying fix.');
            gradient((isnan(gradient))) = 0;
        end
        
        %are we minimizing or maximizing?
        if minimize
            gradient = gradient*-1;
        end

        %update mean gradient
        if isempty( mean_grad )
            mean_grad = 0; %zeros(length(gradient),1);
        end
        mean_grad = (1-(1/ohgd_t))*mean_grad + (1/ohgd_t)*gradient;
        %mean_grad = mean_grad + gradient;
        
        %mean_grad = mean_grad + gradient;
        %update hessian
        if isempty(inv_H)
            inv_H = eye(length(gradient))*lambda0;
        end

        bg = gradient;
        %ohgd.inv_H = (1-(1/ohgd.t))*ohgd.inv_H - ((ohgd.inv_H*bg)*(bg'*ohgd.inv_H))/( 1+ bg'*(ohgd.inv_H*bg));
        %new_inv_H = inv_H - ((inv_H*bg)*(bg'*inv_H))/( 1+ bg'*(inv_H*bg));
        if ohgd_t == 1
            eeps = 1;
            omeps = 1;
        else
            eeps = (1/ohgd_t);
            omeps = 1-eeps;
        end
        
        new_inv_H = (1/eeps)*inv_H - (eeps/omeps)* ((inv_H*bg)*(bg'*inv_H))/(  omeps + eeps*bg'*(inv_H*bg));

        
        if not(all(isfinite(new_inv_H)))
            warning('NaNs encountered during computation of inverse Fisher update');
        else
            inv_H = new_inv_H;
        end
        %update t
        ohgd_t = ohgd_t + 1;

        %return values
        if nargout >= 1
            mg = mean_grad;
        end
        if nargout >=2
            iH = inv_H;
        end
    end

    function [tt, bounded_change] = getParams()
        %have not updated params yet
        if isempty(inv_H)
            tt = [];
            bounded_change = [];
            return;
        end
        
        if isempty(mean_inv_H)
            lc_mean_inv_H = inv_H;
        else
            lc_mean_inv_H = mean_inv_H;
        end
        if average_invH_across_itr
            lc_mean_inv_H = (1-(1/itr))*lc_mean_inv_H + (1/itr)*inv_H  ;
        else
            lc_mean_inv_H = inv_H;
        end
        %first we obtain change to apply
        delta_theta = multiplier*lc_mean_inv_H * mean_grad;

        %constrain if necessary

        %set the signs
        
        sign_grads = sign(delta_theta);

        if auto_gradient_bound
            %get the distance to the upper and lower bound
            %size(ohgd.theta_t)
            %size(ohgd.lower_bounds)

            dL = abs(theta_t - lower_bounds)*limit_per;
            dU = abs(theta_t - upper_bounds)*limit_per;

            %use upperbound if positive sign, lower bound if negative sign
            agradient_bound_array = zeros(size(delta_theta));
            for i=1:length(delta_theta)
                if sign(delta_theta(i)) > 0
                    agradient_bound_array(i) = dU(i);
                elseif sign(delta_theta(i)) < 0
                    agradient_bound_array(i) = dL(i);
                end
            end
            
            %bounded_change = agradient_bound_array;
            bounded_change = sign_grads .* min(abs(agradient_bound_array), gradient_bound_array);
            
        else

        %bound the change
        %delta_theta
        %ohgd.gradient_bound_array
            bounded_change = sign_grads .* min(abs(delta_theta), gradient_bound_array);
        end

        if any(isnan(bounded_change))
           warning('NaN value in bounded change'); 
        end
        
        %apply
        lc_theta_t = theta_t + bounded_change;

        lc_mean_theta = (1-(1/itr))*mean_theta + (1/itr)*lc_theta_t ;

        tt = lc_mean_theta;        
    end

    function [tt] = updateParams(rG, rH)
        %UPDATEPARAMS updates the parameters using the stored hessian
        %rG = reset gradient, rH = reset Fisher

        %obtain mean hessian
        if isempty(mean_inv_H)
            mean_inv_H = inv_H;
        end
        if average_invH_across_itr
            mean_inv_H = (1-(1/itr))*mean_inv_H + (1/itr)*inv_H  ;
        else
            mean_inv_H = inv_H;
        end
        %first we obtain change to apply
        delta_theta = multiplier*mean_inv_H * mean_grad;

        %constrain if necessary

        %set the signs
        sign_grads = sign(delta_theta);

        if auto_gradient_bound
            %get the distance to the upper and lower bound
            %size(ohgd.theta_t)
            %size(ohgd.lower_bounds)

            dL = abs(theta_t - lower_bounds)*limit_per;
            dU = abs(theta_t - upper_bounds)*limit_per;

            %use upperbound if positive sign, lower bound if negative sign
            agradient_bound_array = zeros(size(delta_theta));
            for i=1:length(delta_theta)
                if sign(delta_theta(i)) > 0
                    agradient_bound_array(i) = dU(i);
                elseif sign(delta_theta(i)) < 0
                    agradient_bound_array(i) = dL(i);
                end
            end

            %bounded_change = agradient_bound_array;
            bounded_change = sign_grads .* min(abs(delta_theta), agradient_bound_array);
            bounded_change = sign_grads .* min(abs(bounded_change), gradient_bound_array);
            
        else

        %bound the change
        %delta_theta
        %ohgd.gradient_bound_array
            bounded_change = sign_grads .* min(abs(delta_theta), gradient_bound_array);
        end

        %apply
        theta_t = theta_t + bounded_change;

        mean_theta = (1-(1/itr))*mean_theta + (1/itr)*theta_t ;


        itr = itr+1;
        tt = mean_theta;
        
        
        %we do not reset unless explicitly stated
        if nargin >= 1
            if rG
                resetMeanGrad();
            end
        end

        if nargin >=2
            if rH
                resetHessian();
            end
        end
        
        if auto_plot
            plotParameters();
        end
        
    end

    function resetMeanGrad()
        %RESETMEANGRAD Resets the mean gradient to zero and averaging
        %time to 1 (basically forget the collected gradients)
        ohgd_t = 1;
        mean_grad = 0;
    end

    function resetHessian()
        %RESETHESSIAN Resets the hessian to eye(N)*lambda0
        if not(isempty(inv_H))
            inv_H = eye(size(inv_H))*lambda0;
        end
    end


    function setGradientBounds(gradient_bounds)
        %SETGRADIENTBOUNDS Sets the gradient bounds for the parameters.
        %Helpful to prevent optimising into areas which can cause
        %numerical difficulties.
        
        auto_gradient_bound = true;
        if ischar(gradient_bounds)
            if strcmpi(gradient_bounds, 'AUTO')
                gradient_bound_array = zeros(length(theta_t),1) + Inf;
            else
               error('Gradient bounds should be AUTO or a numeric array'); 
            end
        elseif isnumeric(gradient_bounds)

            if length(gradient_bounds) == 1
                gradient_bound_array = zeros(length(theta_t),1) + ...
                    gradient_bounds;
            else
                gradient_bound_array = gradient_bounds;
            end
        end


    end

    function setLowerBounds(lb)
        %SETLOWERBOUNDS Sets the lower bounds for the parameters.
        %Helpful to prevent optimising into areas which can cause
        %numerical difficulties.
        lower_bounds = lb;
    end

    function setUpperBounds(ub)
        %SETUPPERBOUNDS Sets the upper bounds for the parameters.
        %Helpful to prevent optimising into areas which can cause
        %numerical difficulties.
        upper_bounds = ub;
    end

    function setBounds(lb, ub)
        %SETBOUNDS Sets the lower and upper bounds for the parameters.
        %Helpful to prevent optimising into areas which can cause
        %numerical difficulties.
        setLowerBounds(lb);
        setUpperBounds(ub);
    end


    
end


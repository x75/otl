function [ psogp ] = addToPSOGP_C( x, y, psogp )
%addToPSOGP Summary of this function goes here
%   Detailed explanation goes here
    kstar = getKStar(x,psogp);
    
    %is this a new psogp?
    if not(isfield(psogp, 'phi'))
        for i=1:length(y)
            psogp.alpha{i} = y(i) / (kstar + psogp.noise);
            psogp.C{i} = -1/(kstar + psogp.noise);
        end
        psogp.Q = 1/kstar;
        psogp.phi{1} = x;
    else
        %nope, let's do a geometric test
        k = getKStarVector(x, psogp);
        
        for cl=1:length(y)
            C = psogp.C{cl};
            alpha = psogp.alpha{cl};

            m = k'*alpha;
            s2 = kstar + (k.'*C*k);

            if (s2 < 1e-12)
                fprintf(1, 's2 stability: %f\n',s2);
                s2 = 1e-12;
            end

            if psogp.type == 'r'
                r = -1.0/(s2 + psogp.noise);
                q = -r*(y - m);
            elseif psogp.type == 'c'
                sx2 = psogp.noise + s2;
                sx = sqrt(sx2);
                z = y(cl).*m./sx;

                Erfz = Erf(z);

                dErfz = 1.0/sqrt(2*pi)*exp(-(z.^2)/2);
                dErfz2 = dErfz.*(-z);

                q{cl} = y(cl)./sx .* (dErfz./Erfz);
                r{cl} = (1/sx2)*(dErfz2./dErfz - (dErfz./Erfz)^2);
                %r = r(1);
            end
        end
        
        Q = psogp.Q;
        ehat = Q*k;
        gamma = kstar - dot(k,ehat);
        
        for cl=1:length(y)
            eta{cl} = 1.0./(1.0 + gamma*r{cl});
        end
        if (gamma < 1e-12 )
            fprintf(1, 'gamma stability: %f\n', gamma);
            gamma = 0;
        end

        if (gamma >= psogp.epsilon*kstar)
            %full update

            %             p = zeros(size(s,1),1);
            %             p(end) = 1;

            %add to bv
            psogp.phi{end+1} = x;

            %update Q
            Q(end+1, end+1) = 0;
            ehat(end+1) = -1;
            ehat = ehat(:); %make ehat a column vector

            Q = Q + (1/gamma)*(ehat*ehat.');
            
            for cl=1:length(y)
                
                alpha = psogp.alpha{cl};
                C = psogp.C{cl};
                alpha(end+1,:) = 0;

                s = [C*k; 1];
                alpha = alpha + (s*q{cl});
                C(end+1, end+1) = 0;
                C = C+r{cl}*(s*s.');
                
                psogp.C{cl} = C;
                psogp.alpha{cl} = alpha;

            end

        else

            %sparse update
            for cl=1:length(y)
                alpha = psogp.alpha{cl};
                C = psogp.C{cl};               
                
                s = C*k +ehat;
           
                alpha = alpha + s*(q{cl}*eta{cl});
    
                C = C+r{cl}*eta{cl}*(s*s.');
                
                psogp.C{cl} = C;
                psogp.alpha{cl} = alpha;
            end      
        end

        psogp.Q = Q;

    end

    %basis vector deletion
    len_phi = length(psogp.phi);
    if (len_phi > psogp.capacity)
        %scores
        for cl = 1:length(y)
            
            alpha = psogp.alpha{cl};
            C = psogp.C{cl}; 
            
            for i = 1:len_phi
                scores(i,cl) = (alpha(i,:)).^2/ (Q(i,i) + C(i,i));
            end
        end

        if psogp.deletion_criteria == 'n'
            for i = 1:len_phi
                score(i) = norm(scores(i,:));
            end
        elseif psogp.deletion_criteria == 'm'
            for i = 1:len_phi
                score(i) = max(scores(i,:));
            end
        end

        [min_val, min_index] = min(score);
        psogp = deleteFromPSOGP_C( min_index, psogp );
    end

end

function erfz = Erf(z)
    erfz = normcdf(z,0,1);
end

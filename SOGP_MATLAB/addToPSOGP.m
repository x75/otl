 function [ psogp ] = addToPSOGP( x, y, psogp )
%addToPSOGP Summary of this function goes here
%   Detailed explanation goes here
    kstar = getKStar(x,psogp);
    
    %is this a new psogp?
    if not(isfield(psogp, 'phi'))
       psogp.alpha = y / (kstar + psogp.noise);
       psogp.C = -1/(kstar + psogp.noise);
       psogp.Q = 1/kstar;
       psogp.phi{1} = x;
    else
        %nope, let's do a geometric test
        C = psogp.C;
        alpha = psogp.alpha;
        Q = psogp.Q;
        
        
        k = getKStarVector(x, psogp);
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
            z = y.*m./sx;
            
            Erfz = Erf(z);
            
            dErfz = 1.0/sqrt(2*pi)*exp(-(z.^2)/2);
            dErfz2 = dErfz.*(-z);
            
            q = y/sx * (dErfz/Erfz);
            r = (1/sx2)*(dErfz2/dErfz - (dErfz/Erfz)^2);
        end
            
        ehat = Q*k;

        gamma = kstar - dot(k,ehat);
        eta = 1.0/(1.0 + gamma*r);
        if (gamma < 1e-12 )
            fprintf(1, 'gamma stability: %f\n', gamma);
            gamma = 0;
        end

        if (gamma >= psogp.epsilon*kstar)  
                        %full update
            s = [C*k; 1];
%             p = zeros(size(s,1),1);
%             p(end) = 1;
            
            %add to bv
            psogp.phi{end+1} = x;
            
            %update Q
            Q(end+1, end+1) = 0;
            ehat(end+1) = -1;
            ehat = ehat(:); %make ehat a column vector

            Q = Q + (1/gamma)*(ehat*ehat.');
            
            alpha(end+1,:) = 0;

            alpha = alpha + (s*q);
            C(end+1, end+1) = 0;
            C = C+r*(s*s.');
        else

            %sparse update
            s = C*k +ehat;

            alpha = alpha + s*(q*eta);

            C = C+r*eta*(s*s.');
        end
        
        psogp.C = C;
        psogp.alpha = alpha;
        psogp.Q = Q;
        
    end
    
    %basis vector deletion
    len_phi = length(psogp.phi);   
    if (len_phi > psogp.capacity)
        %scores
        for i = 1:len_phi
            scores(i,:) = (alpha(i,:)).^2/ (Q(i,i) + C(i,i));
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
        psogp = deleteFromPSOGP( min_index, psogp );
    end
    
 end

 function erfz = Erf(z)
    erfz = normcdf(z,0,1);
 end
 
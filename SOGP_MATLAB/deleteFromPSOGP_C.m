function [ psogp ] = deleteFromPSOGP( min_index, psogp )
%DELETEFROMPSOGP deletes a basis vector from the psogp
    %make local copies
    
    Q = psogp.Q;
    qstar = Q(min_index, min_index);
    Qstar = Q(:, min_index);
    Qstar(min_index) = Qstar(end);
    Qstar = Qstar(1:end-1);
    Qrep = Q(:, end);
    Qrep(min_index) = Qrep(end);
    Q(min_index, :) = Qrep';
    Q(:, min_index) = Qrep;
    
    Q = Q(1:end-1, 1:end-1) - (Qstar*Qstar')/qstar;
    
    for cl=1:psogp.num_classes
        alpha = psogp.alpha{cl};
        C = psogp.C{cl};

        %create our temp vars
        alphastar = alpha(min_index,:);
        alpha(min_index, :) = alpha(end, :); %swap with last row
        alpha = alpha(1:end-1,:);
    
        cstar = C(min_index, min_index);
        Cstar = C(:, min_index);
        Cstar(min_index) = Cstar(end, :);
        Cstar = Cstar(1:end-1);
        Crep = C(:,end);
        Crep(min_index) = Crep(end);
        C(min_index, :) = Crep';
        C(:, min_index) = Crep;
    
        qc = (Qstar + Cstar)/(qstar+cstar);
        for i=1:size(alpha,2)
            alpha(:,i) = alpha(:,i) - alphastar(i)*qc;
        end
    
        C = C(1:end-1,1:end-1) + (Qstar*Qstar')/qstar - ...
            ((Qstar + Cstar)*(Qstar+Cstar)')/(qstar+cstar);
        
        
        psogp.alpha{cl} = alpha;
        psogp.C{cl} = C;
    end
    
    psogp.phi{min_index} = psogp.phi{end};
    psogp.phi = psogp.phi(1:end-1);
    

    psogp.Q = Q;
    
end



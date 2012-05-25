%let's generate some simple test (sine wave)
clear();

data = [0:0.01:2*pi].';
X = data(randperm(size(data,1)));
Y = [ sin(X) + normrnd(0,0.1,size(X)) ...
      cos(X) + normrnd(0,0.1,size(X))];

N = length(X);
N = 100;
%plot(X); hold on; plot(Y,'g');

%initialise the PSOGP
capacity = 10;
noise = 0.1;
epsilon = 1e-6;
kernFunc = @kern_gaussian;
kernParams = [1.0 0.9 0.0];

gp_params = [capacity, noise, epsilon];

%'r' is for regression, 'c' is for classification
psogp = initPSOGP(gp_params, kernFunc, kernParams, 'n');

%for each data item, add it to the PSOGP
y_pred = zeros(N,size(Y,2));
for i=1:N %size(X,1)
   [y_pred(i,:), y_var(i)] = predictPSOGP(X(i), psogp);
   psogp = addToPSOGP(X(i,:), Y(i,:), psogp);
end

%plot each dimension in its own plot
for i=1:2
    figure()
    errorbar(X(1:N), y_pred(1:N,i), y_var(1:N), 'b.'); hold on;
    scatter(X(1:size(y_pred,1)), Y(1:size(y_pred,1),i), 'g+'); 
    bvs = cell2mat(psogp.phi);
    scatter(bvs, zeros(size(bvs)), 'r');
    hold off;
end



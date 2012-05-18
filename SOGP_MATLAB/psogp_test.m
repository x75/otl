%let's generate some simple test (sine wave)
clear();
data = [0:0.01:2*pi].';
X = data(randperm(size(data,1)));
Y = sin(X) + normrnd(0,0.1,size(X));
%plot(X); hold on; plot(Y,'g');

%initialise the PSOGP
capacity = 10;
noise = 0.1;
epsilon = 1e-6;
kernFunc = @kern_gaussian;
kernParams = [1.0 0.9 0.0];

gp_params = [capacity, noise, epsilon];

%'r' is for regression, 'c' is for classification
psogp = initPSOGP(gp_params, kernFunc, kernParams, 'r');

%for each data item, add it to the PSOGP
for i=1:100 %size(X,1)

   [y_pred(i), y_var(i)] = predictPSOGP(X(i), psogp);
   psogp = addToPSOGP(X(i), Y(i), psogp);
end
errorbar(X(1:size(y_pred,2)), y_pred, y_var, 'b.'); hold on;
scatter(X(1:size(y_pred,2)), Y(1:size(y_pred,2)), 'g+'); 




bvs = cell2mat(psogp.phi);
scatter(bvs, zeros(size(bvs)), 'r');
hold off;
%let's generate some simple test (sine wave)
clear();
N = 500;
X = [rand(N,1)*2 - 1 rand(N,1)*2 - 1];
Y = sign(X(:,1) + X(:,2))

%plot(X); hold on; plot(Y,'g');

%initialise the PSOGP
capacity = 100;
noise = 0.1;
epsilon = 1e-5;
kernFunc = @kern_gaussian;
kernParams = [1.0 0.001 0.0];

gp_params = [capacity, noise, epsilon];

%'r' is for regression, 'c' is for classification
psogp = initPSOGP(gp_params, kernFunc, kernParams, 'c');

%for each data item, add it to the PSOGP
for i=1:N %size(X,1)
   [y_pred(i), y_var(i)] = predictPSOGP(X(i), psogp);
   psogp = addToPSOGP(X(i), Y(i), psogp);
   size(psogp.phi)
end

wrong_ones = find(y_pred.' ~= Y)
positives = find(y_pred.' == 1);
negatives = find(y_pred.' == -1);
scatter(X(positives,1), X(positives,2) , 'g+'); hold on;
scatter(X(negatives,1), X(negatives,2), 'ko');
scatter(X(wrong_ones,1), X(wrong_ones,2), 'r*');
hold off;
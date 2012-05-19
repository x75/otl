%let's generate some simple test (sine wave)
clear();
N = 1000;

%centres = [2.0 2.0; -2.0 -2.0]

X = [rand(N,1)*2 - 1 rand(N,1)*2 - 1];
Y = (X(:,1).^2 + X(:,2).^2 < 0.5^2).*2 - 1



%plot(X); hold on; plot(Y,'g');

%initialise the PSOGP
capacity = 500;
noise = 1.0;
epsilon = 1e-5;
kernFunc = @kern_gaussian;
kernParams = [0.5 0.001 0.0];

gp_params = [capacity, noise, epsilon];

%'r' is for regression, 'c' is for classification
psogp = initPSOGP(gp_params, kernFunc, kernParams, 'c');

%for each data item, add it to the PSOGP
for i=1:N %size(X,1)
   [y_pred(i), y_var(i)] = predictPSOGP(X(i,:), psogp);
   
   psogp = addToPSOGP(X(i,:), Y(i), psogp);
   size(psogp.phi)
end

wrong_ones = find(y_pred.' ~= Y)
positives = find(Y == 1);
negatives = find(Y == -1);
scatter(X(positives,1), X(positives,2) , 'g+'); hold on;
scatter(X(negatives,1), X(negatives,2), 'ko');
scatter(X(wrong_ones,1), X(wrong_ones,2), 'r*');
error_rate = size(wrong_ones,1)/N
y_var(wrong_ones)

hold off;
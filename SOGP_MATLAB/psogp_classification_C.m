%let's generate some simple test (sine wave)
clear();
N = 3000;

%centres = [2.0 2.0; -2.0 -2.0]
rand('twister',6657);
X = [rand(N,1)*2 - 1 rand(N,1)*2 - 1];
Y = [((X(:,1).^2 + X(:,2).^2 < 0.5^2).*2 - 1) ...
    (sign((X(:,1) + X(:,2)))) ...
    (sign((X(:,1) + X(:,2).^3))) ...
    (sign( (0.1*X(:,1).^3 + X(:,2).^4 + 0.2*(X(:,1).*X(:,2))))) ...
    ];


%plot(X); hold on; plot(Y,'g');

%initialise the PSOGP
capacity = 300;
noise = 1.0;
epsilon = 1e-5;
kernFunc = @kern_gaussian;
kernParams = [0.1 0.001 0.0];

gp_params = [capacity, noise, epsilon];

%'r' is for regression, 'c' is for classification
psogp = initPSOGP_C(gp_params, kernFunc, kernParams, 'c', 'n', 4);

%for each data item, add it to the PSOGP
y_pred = zeros(size(Y));
y_var = zeros(size(Y));
tic()
for i=1:N %size(X,1)
   [y_pred(i,:), y_var(i,:)] = predictPSOGP_C(X(i,:), psogp);
   
   psogp = addToPSOGP_C(X(i,:), Y(i,:), psogp);

end
stop_time = toc()
for i = 1:size(y_pred,2)
    figure()
    wrong_ones = find(y_pred(:,i) ~= Y(:,i));
    positives = find(Y(:,i) == 1);
    negatives = find(Y(:,i) == -1);
    scatter(X(positives,1), X(positives,2) , 'g+'); hold on;
    scatter(X(negatives,1), X(negatives,2), 'ko');
    scatter(X(wrong_ones,1), X(wrong_ones,2), 'r*');
    error_rate = size(wrong_ones,1)/N
    y_var(wrong_ones);
end

fps = N/stop_time

hold off;
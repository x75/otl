%let's generate some simple test (sine wave)
clear();
data = load('data/image_dataset.test');
X = data(:,2:end);

N = length(X);

Y_data = data(:,1);
rand('twister',6657);
for i=1:size(X,2)
   X(:,i) = X(:,i) - mean(X(:,i)); 
    
   num_range = (max(X(:,i)) - min(X(:,i)));
   %num_range = var(X(:,i));
   if num_range == 0
       num_range = 1;
   end
   X(:,i) = (X(:,i) - min(X(:,i))) / num_range; 
end

Y = zeros(length(Y_data), max(Y_data)+1) - 1;
for i=1:size(Y_data,1)
    Y(i, Y_data(i) + 1) = 1; 
end

rp = randperm(length(X));
X = X(rp,:);
Y = Y(rp,:);
Y_data = Y_data(rp) + 1;

%plot(X); hold on; plot(Y,'g');

%initialise the PSOGP
capacity = 300;
noise = 1.0;
epsilon = 1e-5;
kernFunc = @kern_gaussian;
kernParams = [0.25 0.001 0.0];

gp_params = [capacity, noise, epsilon];

%'r' is for regression, 'c' is for classification
psogp = initPSOGP_C(gp_params, kernFunc, kernParams, 'c', 'm', 7);

%for each data item, add it to the PSOGP
y_pred = zeros(size(Y));
y_var = zeros(size(Y));
tic()
predictions = zeros(N,1);
for i=1:N %size(X,1)
   [y_pred(i,:), y_var(i,:)] = predictPSOGP_C(X(i,:), psogp);
   [pred_var(i), predictions(i)] = max(y_var(i,:));
   psogp = addToPSOGP_C(X(i,:), Y(i,:), psogp);

end
toc()

wrong_ones = find(predictions ~= Y_data(1:N))
pred_var(wrong_ones)
error_rate = size(wrong_ones,1)/N
accuracy = 1-error_rate

conf_levels = [0:0.01:1.0];
for i=1:length(conf_levels)
    confident_ones = find(pred_var >= conf_levels(i));
    if isempty(confident_ones)
        continue;
    end
    wrong_ones = find(predictions(confident_ones) ~= Y_data(confident_ones));
    error_rate = size(wrong_ones,1)/N;
    accuracy = 1-error_rate;
    
    accuracies(i) = accuracy;
    num_classified(i) = length(confident_ones)/N;
end
figure()
subplot(2,1,1);
plot(conf_levels,accuracies, 'r-'); ylabel('Accuracies');
xlabel('Classification Threshold');
subplot(2,1,2);
plot(conf_levels,num_classified, 'b-'); ylabel('Number Classified');
xlabel('Classification Threshold');


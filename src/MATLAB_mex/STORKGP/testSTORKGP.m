%STORKGP Example MATLAB script
% This sample script shows how to use the STORKGP MATLAB class. Remember,
% first you have to compile STORKGP using the compileSTORKGP_mex.m script.
% Edit it to point to the correct directories (if necessary) and run it. 
%
% In this example, we'll just create a simple combination of sine and cos 
% waves with some noise and learn to predict it in an online fashion. 

clear();

%let's create some simple data

data_raw = dlmread('context_data/learn_data_2_last_state.csv', ';');
data_raw = data_raw(:,[2,3,4,5,8,9,10,11]);
data_range = max(data_raw) - min(data_raw);
data = data_raw ./ repmat(data_range, size(data_raw,1), 1);

x_data = data(:,3:end);
y_data = data(:,1:2);

N = size(x_data,1);

%some linear combination of the two inputs (plus noise and 
% future_step's into the future)
%y_data = [x_data(:,1) + x_data(:,2), 0.5*x_data(:,2) - 0.5*x_data(:,1)];
%y_data = y_data(future_step:end,: ); 
%y_data_obs = y_data + normrnd(0,0.05,size(y_data));

%Create out storkgp object and store the handle
input_dim = 6;
output_dim = 2;
tau = 5; %how long is the method's memory
l = 1.0; %characteristic lengthscale of the GP
rho = 0.99; %spectral radius (usually 0.9 to 0.99)
alpha = 1.0; %scalar multipler for the kernel (usually 1.0)
noise = 0.05; %noise in the system
epsilon = 1e-4; %epsilon parameter for updates (small e.g. 1e-4)
                %set this larger (e.g. 1e-3 or 1e-2) if you encounter
                %numerical errors
capacity = 100; %capacity of the model (how many basis vectors you 
                %want to store). Larger models will be slower. 

kernParams = [l, rho, alpha];

%create our STORKGP object
storkgp = STORKGP(input_dim, output_dim, tau, ...
    kernParams, noise, epsilon, capacity );

%loop through our data, predict and learn.
%inputs and outputs should be in *row* vector format.
tic
for i=1:N 
    i
    %update model with the input 
    storkgp.update(x_data(i,:));
    
    %get the prediction
    %[pred_mean(i,:), pred_var(i,:)] = storkgp.predict();
    
    %compute the error (with the TRUE y data)
    %error(i) = norm(pred_mean(i,:) - y_data(i,:));
    
    %train the model with the observation
    storkgp.train(y_data(i,:));

end
toc
storkgp.resetState();
tic
for i=1:N 
    %update model with the input 
    i
    storkgp.update(x_data(i,:));
    
    %get the prediction
    [pred_mean(i,:), pred_var(i,:)] = storkgp.predict();
    
    %compute the error (with the TRUE y data)
    error(i) = norm(pred_mean(i,:) - y_data(i,:));
    
    %train the model with the observation
    %storkgp.train(y_data(i,:));

end
toc
%if you want to RESET the memory state of the model, you can use:
%storkgp.resetState(); %this does not reset the model, just the "memory".

%if you want to save the model
storkgp.save('storkgptest');

%plots
figure();
subplot(3,1,1);
plot(y_data(1:end,1), 'b-');hold on;
plot(pred_mean(:,1), 'g-'); 
legend('true y','observed y', 'prediction');
ylabel('Y(1)');
subplot(3,1,2);
plot(y_data(1:end,2), 'b-');hold on;
plot(y_data(:,2), 'k+');
plot(pred_mean(:,2), 'g-'); 
ylabel('Y(2)');
subplot(3,1,3);
plot(error, 'r');
ylabel('Errors');
xlabel('Time');


target = data(9:end, 3:4);
estimate = data_raw(1:end-8, 3:4) + pred_mean(1:end-8, :).*repmat(data_range(1:2),size(data_raw,1)-8,1);
estimate = estimate./repmat(data_range(3:4),size(data_raw,1)-8,1);

figure
subplot(2,1,1)
plot(target(:,1), 'r'), hold on
plot(estimate(:,1), 'b')
subplot(2,1,2)
plot(target(:,2), 'r'), hold on
plot(estimate(:,2), 'b')

naive_error =0.0559;

rmse = mean(sqrt(sum((target-estimate).^2,2)))
norm_rmse = 1-rmse/naive_error

%destroy the object we created, using clear() would do it too
delete(storkgp)


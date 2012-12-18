%STORKGP Example MATLAB script
% This sample script shows how to use the STORKGP MATLAB class. Remember,
% first you have to compile STORKGP using the compileSTORKGP_mex.m script.
% Edit it to point to the correct directories (if necessary) and run it.
%
% In this example, we'll just create a simple combination of sine and cos
% waves with some noise and learn to predict it in an online fashion.

clear();

%let's create some simple data
N = 500;    % 500 time steps
future_step = 2; %how many steps into the future should we predict? 
x_data = [sin([0:N]*0.1).' 0.2*cos([0:N]*0.5).'];

%some linear combination of the two inputs (plus noise and 
% future_step's into the future)
y_data = [x_data(:,1) + x_data(:,2), 0.5*x_data(:,2) - 0.5*x_data(:,1)];
y_data = y_data(future_step:end,: ); 
y_data_obs = y_data + normrnd(0,0.05,size(y_data));

%Create out storkgp object and store the handle
input_dim = 2;
output_dim = 2;
tau = 3; %how long is the method's memory
l = 1.0; %characteristic lengthscale of the GP
rho = 0.99; %spectral radius (usually 0.9 to 0.99)
alpha = 1.0; %scalar multipler for the kernel (usually 1.0)
noise = 0.1; %noise in the system
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
for i=1:N-future_step 
    %update model with the input 
    storkgp.update(x_data(i,:));
    
    %get the prediction
    [pred_mean(i,:), pred_var(i,:)] = storkgp.predict();
    
    %compute the error (with the TRUE y data)
    error(i) = norm(pred_mean(i,:) - y_data(i,:));
    
    %train the model with the observation
    storkgp.train(y_data_obs(i,:));

end

%if you want to RESET the memory state of the model, you can use:
%storkgp.resetState(); %this does not reset the model, just the "memory".

%if you want to save the model
storkgp.save('storkgptest');

%plots
figure();
subplot(3,1,1);
plot(y_data(1:end,1), 'b-');hold on;
plot(y_data_obs(:,1), 'k+'); 
plot(pred_mean(:,1), 'g-'); 
legend('true y','observed y', 'prediction');
ylabel('Y(1)');
subplot(3,1,2);
plot(y_data(1:end,2), 'b-');hold on;
plot(y_data_obs(:,2), 'k+');
plot(pred_mean(:,2), 'g-'); 
ylabel('Y(2)');
subplot(3,1,3);
plot(error, 'r');
ylabel('Errors');
xlabel('Time');


%destroy the object we created, using clear() would do it too
delete(storkgp)

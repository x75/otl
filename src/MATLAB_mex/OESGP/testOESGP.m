%OESGP Example MATLAB script
% This sample script shows how to use the OESGP MATLAB class. Remember,
% first you have to compile OESGP using the compileOESGP_mex.m script.
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

%Create out oesgp object and store the handle
input_dim = 2;
output_dim = 2;

reservoir_size = 100; %the reservoir size?
input_weight = 1.0; %input weight, usually 1.0
output_feedback_weight = 0; %usually, we don't use output feedback
activation_function = 'TANH'; %tanh activation function
leak_rate = 0.0; %no leak rate here, (between 0 and 1.0)
connectivity = 0.1; %connectivity of the reservoir
spectral_radius = 0.99; %spectral radius (typically 0.9 - 0.99)
use_inputs_in_state= false; %use the inputs directly? 
random_seed = 10; %change the seed used to randomly create the reservoir

l = 1.0; %characteristic lengthscale of the GP
alpha = 1.0; %scalar multipler for the kernel (usually 1.0)
noise = 0.1; %noise in the system
epsilon = 1e-4; %epsilon parameter for updates (small e.g. 1e-4)
                %set this larger (e.g. 1e-3 or 1e-2) if you encounter
                %numerical errors
capacity = 100; %capacity of the model (how many basis vectors you 
                %want to store). Larger models will be slower. 

kernel_parameters = [l, alpha];

%create our OESGP object
oesgp = OESGP(input_dim, output_dim, reservoir_size,...
                    input_weight, output_feedback_weight,...
                    activation_function,...
                    leak_rate,...
                    connectivity, spectral_radius,...
                    use_inputs_in_state,...
                    kernel_parameters,...
                    noise, epsilon, capacity, random_seed );

%loop through our data, predict and learn.
%inputs and outputs should be in *row* vector format.
for i=1:N-future_step 
    %update model with the input 
    oesgp.update(x_data(i,:));
    
    %get the prediction
    [pred_mean(i,:), pred_var(i,:)] = oesgp.predict();
    
    %compute the error (with the TRUE y data)
    error(i) = norm(pred_mean(i,:) - y_data(i,:));
    
    %train the model with the observation
    oesgp.train(y_data_obs(i,:));

end

%if you want to RESET the memory state of the model, you can use:
%oesgp.resetState(); %this does not reset the model, just the "memory".

%if you want to save the model
oesgp.save('oesgptest');

%loading a saved model is easy:
%oesgp.load('oesgptest');

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
delete(oesgp)


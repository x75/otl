/**
  OESGP Sin Wave prediction Example Code

  Copyright 2012 All rights reserved. Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com

    See LICENSE.txt for license information.
  **/

#include "otl.h"
#include "otl_oesgp.h"

#include <iostream>
#include <cmath>

using namespace OTL;
using namespace std;


int main(void) {

    //Create our OESGP object
    OESGP oesgp;

    //Problem parameters
    //we want to predict a simple sine wave
    int input_dim = 1;
    int output_dim = 1;

    //Reservoir Parameters
    //you can change these to see how it affects the predictions
    int reservoir_size = 100;
    double input_weight = 1.0;
    double output_feedback_weight = 0.0;
    int activation_function = Reservoir::TANH;
    double leak_rate = 0.9;
    double connectivity = 0.1;
    double spectral_radius = 0.90;
    bool use_inputs_in_state = false;

    VectorXd kernel_parameters(2); //gaussian kernel parameters
    kernel_parameters << 1.0, 1.0; //l = 1.0, alpha = 1.0

    //SOGP parameters
    double noise = 0.01;
    double epsilon = 1e-3;
    int capacity = 200;

    int random_seed = 0;

    try {
        //Initialise our OESGP
        oesgp.init( input_dim, output_dim, reservoir_size,
                    input_weight, output_feedback_weight,
                    activation_function,
                    leak_rate,
                    connectivity, spectral_radius,
                    use_inputs_in_state,
                    kernel_parameters,
                    noise, epsilon, capacity, random_seed);


        //now we loop using a sine wave
        unsigned int max_itr = 1000;

        //note that we use Eigen VectorXd objects
        //look at http://eigen.tuxfamily.org for more information about Eigen
        VectorXd input(1);  //the 1 is the size of the vector
        VectorXd output(1);

        VectorXd state;
        VectorXd prediction;
        VectorXd prediction_variance;

        for (unsigned int i=0; i<max_itr; i++) {

            //create the input and output
            input(0) = sin(i*0.01);
            output(0) = sin((i+1)*0.01);

            //update the OESGP with the input
            oesgp.update(input);

            //predict the next state
            oesgp.predict(prediction, prediction_variance);

            //print the error
            double error = (prediction - output).norm();
            cout << "Error: " << error << ", |BV|: "
                 << oesgp.getCurrentSize() <<  endl;

            //train with the true next state
            oesgp.train(output);
        }

        cout << "Testing saving and loading model " << std::endl;

        //Here, we demonstrate how to save and load a model
        //we save the original model and create a new oesgp object to load the
        //saved model into

        oesgp.save("oesgptest");

        OESGP oesgp2;
        oesgp2.load("oesgptest");

        //prediction test
        for (unsigned int i=max_itr; i<max_itr+50; i++) {
            input(0) = sin(i*0.01);
            output(0) = sin((i+1)*0.01);

            //update
            oesgp2.update(input);

            //predict
            oesgp2.predict(prediction, prediction_variance);
            double error = (prediction - output).norm();
            cout << "Error: " << error << endl;
        }

    } catch (OTLException &e) {
        e.showError();
    }
    return 0;
}

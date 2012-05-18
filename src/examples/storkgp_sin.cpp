/**
  STORKGP Sin Wave prediction Example Code

  Copyright 2012 All rights reserved. Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com

    See LICENSE.txt for license information.
  **/

#include "otl.h"
#include "otl_storkgp.h"

#include <iostream>
#include <cmath>

using namespace OTL;
using namespace std;


int main(void) {
	//create our STORKGP object
	STORKGP storkgp;

    unsigned int input_dim = 1;
    unsigned int output_dim = 1;

	//parameters for the STORKGP algorithm
    unsigned int tau = 20;	//length of memory

	//kernel parameters
    double l = 0.5;
    double rho = 0.99;
    double alpha = 1.0;

	//SOGP parameters
    double noise = 0.0001;
    double epsilon = 1e-4;
    unsigned int capacity = 100;

    VectorXd kernel_parameters(4);
    //[l rho alpha input_dim]
    kernel_parameters << l, rho, alpha, input_dim;


    try {
        //Initialise our STORKGP
	    storkgp.init(input_dim, output_dim,
	                 tau,
	                 STORKGP::RECURSIVE_GAUSSIAN,
	                 kernel_parameters,
	                 noise, epsilon, capacity);

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

            //update the STORKGP with the input
            storkgp.update(input);

            //predict the next state
            storkgp.predict(prediction, prediction_variance);

            //print the error
            double error = (prediction - output).norm();
            cout << "Error: " << error << ", |BV|: "
                 << storkgp.getCurrentSize() <<  endl;

            //train with the true next state
            storkgp.train(output);
        }

        cout << "Testing saving and loading model " << std::endl;

        //Here, we demonstrate how to save and load a model
        //we save the original model and create a new storkgp object to load the
        //saved model into

        storkgp.save("storkgptest");

        STORKGP storkgp2;
        storkgp2.load("storkgptest");

        //prediction test
        for (unsigned int i=max_itr; i<max_itr+50; i++) {
            input(0) = sin(i*0.01);
            output(0) = sin((i+1)*0.01);

            //update
            storkgp2.update(input);

            //predict
            storkgp2.predict(prediction, prediction_variance);
            double error = (prediction - output).norm();
            cout << "Error: " << error << endl;
        }

    } catch (OTLException &e) {
        e.showError();
    }
    return 0;
}

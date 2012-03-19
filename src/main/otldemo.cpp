/**
  Online Temporal Learning (OTL) Demo program
  Copyright 2012 Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com

  This code is free for use in non-commercial settings: at academic institutions, teaching institutes,
  research purposes and non-profit research. Also free for use by hobbyists and students who are not
  making money from it.

  If you use this code for commercial use, you must contact the author for commercial
  licenses, i.e., if you make some money out of this software, I think it's fair that I get some of it
  so that I can support my family and the causes I believe in.

  As usual, no warranties are implied nor guaranteed. Use at your own risk.

  **/

#include "otl.h"
#include "otl_window.h"
#include "otl_rls.h"
#include "otl_reservoir.h"
#include "otl_kernel_gaussian.h"
#include <iostream>
#include <cmath>

using namespace OTL;
using namespace std;

void gaussianKernelTest(void) {

    try {
        GaussianKernel gk;

        VectorXd param(2);
        double l = 10;
        double alpha = 1.0;
        param(0) = l;
        param(1) = alpha;

        gk.init(10, param);

        VectorXd x(10);
        VectorXd y(10);
        x << 1,2,3,4,5,6,7,8,9,10;
        y = x;
        y = x*2;

        std::cout << gk.eval(x,x) << std::endl;
        std::cout << gk.eval(x,y) << std::endl;

        double eps = 1e-12;
        if (gk.eval(x,y) - 0.145876 > eps) {
            std::cout << "Gaussian Kernel produced a WRONG answer!!! " << std::endl;
        } else {
            std::cout << "Gaussian Kernel produced a RIGHT answer!!! " << std::endl;
        }

    } catch (OTLException &e) {
        e.showError();
    }

}


void sinTestRLSESN(void) {
    //let's create our reservoir
    Reservoir res;

    unsigned int input_dim = 1;
    unsigned int output_dim = 1;
    unsigned int res_size = 50;
    int activation_func = Reservoir::TANH;
    double input_weight = 1.0;
    double output_feedback_weight = 0.0;
    double leak_rate = 0.9;
    double spectral_radius = 0.99;
    double connectivity = 0.1;
    bool use_inputs_in_state = true;
    int seed = 0;

    res.init(input_dim, output_dim,res_size,
             input_weight, output_feedback_weight,
             activation_func,
             leak_rate, connectivity, spectral_radius,
             use_inputs_in_state,
             seed);

    //let's create our learning algorithm
    unsigned int state_dim = res.getStateSize();
    double delta = 0.1;
    double lambda = 0.99;
    double noise = 1e-12;
    RLS rls;
    rls.init(state_dim, 1.0, delta, lambda, noise);

    //now we loop using a sine wave
    unsigned int max_itr = 2000;
    VectorXd input(1);
    VectorXd output(1);

    VectorXd state;
    VectorXd prediction;
    VectorXd prediction_variance;

    for (unsigned int i=0; i<max_itr; i++) {
        input(0) = sin(i*0.01);
        output(0) = sin((i+1)*0.01);

        //update
        res.update(input);

        //predict
        res.getState(state);
        rls.predict(state, prediction, prediction_variance);
        double error = (prediction - output).norm();
        cout << "Error: " << error << endl;

        //train
        rls.train(state, output);
    }

    cout << "Testing saving and loading model and reservoir" << std::endl;
    try {
        Reservoir res2;
        res.save("restest.feat");
        res2.load("restest.feat");

        RLS rls2;
        rls.save("rlstest.model");
        rls2.load("rlstest.model");

        for (unsigned int i=max_itr; i<max_itr+50; i++) {
            input(0) = sin(i*0.01);
            output(0) = sin((i+1)*0.01);

            //update
            res2.update(input);

            //predict
            res2.getState(state);
            rls2.predict(state, prediction, prediction_variance);
            double error = (prediction - output).norm();
            cout << "Error: " << error << endl;
        }

    } catch (OTLException &e) {
        e.showError();
    }

}

void sinTestWRLS(void) {
    //let's create our window
    Window delay_window;
    delay_window.init(1, 1, 10);

    //let's create our learning algorithm
    unsigned int state_dim = delay_window.getStateSize();
    double delta = 0.1;
    double lambda = 0.90;
    double noise = 1e-12;
    RLS rls;
    rls.init(state_dim, 1.0, delta, lambda, noise);

    //now we loop using a sine wave
    unsigned int max_itr = 1000;
    VectorXd input(1);
    VectorXd output(1);

    VectorXd state;
    VectorXd prediction;
    VectorXd prediction_variance;

    for (unsigned int i=0; i<max_itr; i++) {
        input(0) = sin(i*0.01);
        output(0) = sin((i+1)*0.01);

        //update
        delay_window.update(input);

        //predict
        delay_window.getState(state);
        rls.predict(state, prediction, prediction_variance);
        double error = (prediction - output).norm();
        cout << "Error: " << error << endl;


        //train
        rls.train(state, output);
    }

    cout << "Testing saving and loading model " << std::endl;
    try {
        RLS rls2;
        rls.save("rlstest.model");
        rls2.load("rlstest.model");

        for (unsigned int i=max_itr; i<max_itr+50; i++) {
            input(0) = sin(i*0.01);
            output(0) = sin((i+1)*0.01);

            //update
            delay_window.update(input);

            //predict
            delay_window.getState(state);
            rls2.predict(state, prediction, prediction_variance);
            double error = (prediction - output).norm();
            cout << "Error: " << error << endl;
        }

    } catch (OTLException &e) {
        e.showError();
    }




}

int main(int argc, char **argv) {
    gaussianKernelTest();
    return 0;

    sinTestRLSESN();
    return 0;

    sinTestWRLS();
    return 0;
    //some code here
    Window delay_window;
    delay_window.init(1,1, 10);

    VectorXd state;

    //update our window to test
    for (unsigned int i=0; i<20; i++) {
        VectorXd input(1);
        input(0) = i;
        delay_window.update(input);
    }

    delay_window.getState(state);
    std::cout << "State: " << endl;
    cout << state << endl;

    //save and reload
    delay_window.save("test_window.data");

    Window test_window;
    test_window.load("test_window.data");
    VectorXd loaded_state;
    delay_window.getState(loaded_state);

    std::cout << "Loaded State: " << endl;
    cout << loaded_state << endl;

    return 0;
}

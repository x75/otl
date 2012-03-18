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
#include <iostream>
#include <cmath>

using namespace OTL;
using namespace std;


void sinTestWRLS(void) {
    //let's create our window
    Window delay_window;
    delay_window.init(1, 1, 10);

    //let's create our learning algorithm
    unsigned int state_dim = delay_window.getStateSize();
    double delta = 0.1;
    double lambda = 0.99;
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

}

int main(int argc, char **argv) {
    sinTestWRLS();
    return 0;
    //some code here
    Window delay_window;
    delay_window.init(1,1, 10);

    VectorXd state;

    //update our window to test
    for (unsigned int i=0; i<5; i++) {
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

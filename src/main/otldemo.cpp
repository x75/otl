/**
  Online Temporal Learning (OTL) Demo program
  Copyright 2012 All rights reserved. Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com

    See LICENSE.txt for license information.
  **/

#include "otl.h"

#include "otl_rls.h"
#include "otl_sogp.h"

#include "otl_reservoir.h"
#include "otl_window.h"
#include "otl_kernel_gaussian.h"
#include "otl_kernel_factory.h"

#include "otl_oesgp.h"
#include "otl_storkgp.h"

#include <iostream>
#include <cmath>

using namespace OTL;
using namespace std;

void oesgpTest(void) {
    OESGP oesgp;

    int input_dim = 1;
    int output_dim = 1;
    int reservoir_size = 100;
    double input_weight = 1.0;
    double output_feedback_weight = 0.0;
    int activation_function = Reservoir::TANH;
    double leak_rate = 0.9;
    double connectivity = 0.1;
    double spectral_radius = 0.90;
    bool use_inputs_in_state = false;
    VectorXd kernel_parameters(2);
    kernel_parameters << 1.0, 1.0;
    double noise = 0.01;
    double epsilon = 1e-3;
    int capacity = 200;
    int random_seed = 0;

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
    VectorXd input(1);
    VectorXd output(1);

    VectorXd state;
    VectorXd prediction;
    VectorXd prediction_variance;

    for (unsigned int i=0; i<max_itr; i++) {
        input(0) = sin(i*0.01);
        output(0) = sin((i+1)*0.01);

        //update
        oesgp.update(input);

        //predict
        oesgp.predict(prediction, prediction_variance);
        double error = (prediction - output).norm();
        cout << "Error: " << error << ", |BV|: " << oesgp.getCurrentSize() <<  endl;


        //train
        oesgp.train(output);
    }

    cout << "Testing saving and loading model " << std::endl;
    try {
        OESGP oesgp2;
        oesgp.save("oesgptest");
        oesgp2.load("oesgptest");

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

}

void sinTestSTORKGP(void) {

    int input_dim = 1;
    int output_dim = 1;
    unsigned int tau = 20;

    double noise = 0.0001;
    double epsilon = 1e-4;
    unsigned int capacity = 100;

    double l = 0.5;
    double rho = 0.99;
    double alpha = 1.0;

    VectorXd kernel_parameters(4);
    //[l rho alpha input_dim]
    kernel_parameters << l, rho, alpha, input_dim;

    STORKGP storkgp;
    storkgp.init(input_dim, output_dim,
                 tau,
                 STORKGP::RECURSIVE_GAUSSIAN,
                 kernel_parameters,
                 noise, epsilon, capacity);

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
        storkgp.update(input);

        //predict
        storkgp.predict(prediction, prediction_variance);
        double error = (prediction - output).norm();
        cout << "Error: " << error << ", |BV|: " << storkgp.getCurrentSize() <<  endl;


        //train
        storkgp.train(output);
    }

    cout << "Testing saving and loading model " << std::endl;
    try {
        STORKGP storkgp2;
        storkgp.save("storkgptest");
        storkgp2.load("storkgptest");

        for (unsigned int i=max_itr; i<max_itr+100; i++) {
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
}


void storkgpValgrindTest(void) {
    for (unsigned int i=0; i<100; i++) {
        sinTestSTORKGP();
    }
}


void sinTestSOGPWin(void) {
    //let's create our window
    Window delay_window;
    delay_window.init(1, 1, 10);

    //let's create our learning algorithm
    unsigned int state_dim = delay_window.getStateSize();

    double l = 1.0;
    //double rho = 0.99;
    double alpha = 0.5;
    //double input_dim = 1;

    VectorXd params(2);
    //[l rho alpha input_dim]
    params << l, alpha;

    GaussianKernel rgk;
    rgk.init(state_dim, params);

    //create out kernel factory
    KernelFactory kern_factory;
    initKernelFactory(kern_factory);

    SOGP sogp;
    double noise = 0.0001;
    double epsilon = 1e-3;
    unsigned int capacity = 100;
    unsigned int output_dim = 1;

    sogp.init(state_dim, output_dim, rgk, noise, epsilon, capacity);

    //now we loop using a sine wave
    unsigned int max_itr = 100;
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
        sogp.predict(state, prediction, prediction_variance);
        double error = (prediction - output).norm();
        cout << "Error: " << error << ", |BV|: " << sogp.getCurrentSize() <<  endl;


        //train
        sogp.train(state, output);
    }


    cout << "Testing saving and loading model " << std::endl;

    try {
        SOGP sogp2;
        sogp.save("sogptest.model");
        //sogp2.setKernelFactory(kern_factory);
        sogp2.load("sogptest.model");

        for (unsigned int i=max_itr; i<max_itr+50; i++) {
            input(0) = sin(i*0.01);
            output(0) = sin((i+1)*0.01);

            //update
            delay_window.update(input);

            //predict
            delay_window.getState(state);
            sogp2.predict(state, prediction, prediction_variance);
            double error = (prediction - output).norm();
            cout << "Error: " << error << endl;
        }
    return;
    } catch (OTLException &e) {
        e.showError();
    }
}

void SOGPMultidimTest(void) {
    //Create our gaussian kernel
    GaussianKernel gk;
    unsigned int output_dim = 2;

    VectorXd param(2);
    double l = 1.0;
    double alpha = 1.0;
    param(0) = l;
    param(1) = alpha;
    unsigned int state_dim = 2;
    gk.init(state_dim, param);

    //create out kernel factory
    KernelFactory kern_factory;
    initKernelFactory(kern_factory);

    SOGP sogp;
    double noise = 0.0001;
    double epsilon = 1e-4;
    unsigned int capacity = 20;


    try {
        sogp.init(state_dim, output_dim, gk, noise, epsilon, capacity);

        //now we loop using a sine wave
        unsigned int max_itr = 100;
        VectorXd input(2);
        VectorXd output(2);

        VectorXd state;
        VectorXd prediction;
        VectorXd prediction_variance;

        VectorXd all_input = VectorXd::Random(max_itr)*2*M_PI;
        //std::cout << all_input << std::endl;
        for (unsigned int i=0; i<max_itr; i++) {
            input(0) = all_input(i);//sin(i*0.01);
            input(1) = all_input(i);

            output(0) = sin(input(0));//sin((i+1)*0.01);
            output(1) = cos(input(0));
            //predict
            sogp.predict(input, prediction, prediction_variance);
            //std::cout << prediction << std::endl;
            VectorXd error = (prediction - output).array() * (prediction- output).array();
            for (unsigned int j=0; j<error.rows(); j++) error(j) = sqrt(error(j));
            cout << "Error: " << error.transpose() << ", " << sogp.getCurrentSize() << endl;

            //train
            sogp.train(input, output);

        }


        sogp.save("sogp.model");
        SOGP sogp2;
        sogp2.setKernelFactory(kern_factory);
        sogp2.load("sogp.model");
        std::cout << "\nTesting save and load facilities\n" << std::endl;
        for (unsigned int i=0; i<max_itr; i++) {
            input(0) = all_input(i);//sin(i*0.01);
            input(1) = all_input(i);

            output(0) = sin(input(0));//sin((i+1)*0.01);
            output(1) = cos(input(0));
            //predict
            sogp2.predict(input, prediction, prediction_variance);
            //std::cout << prediction << std::endl;
            VectorXd error = (prediction - output).array() * (prediction- output).array();
            for (unsigned int j=0; j<error.rows(); j++) error(j) = sqrt(error(j));
            cout << "Error: " << error.transpose() << ", " << sogp2.getCurrentSize() << endl;

        }


    } catch (OTLException &e) {
        e.showError();
    }

}


void SOGPTest(void) {
    //Create our gaussian kernel
    GaussianKernel gk;
    unsigned int output_dim = 1;
    double noise = 0.0001;
    double epsilon = 1e-4;
    unsigned int capacity = 20;

    VectorXd param(2);
    double l = 1.0;
    double alpha = 1.0;
    param(0) = l;
    param(1) = alpha;
    unsigned int state_dim = 1;
    gk.init(1, param);

    //create out kernel factory
    KernelFactory kern_factory;
    initKernelFactory(kern_factory);


    SOGP sogp;
    try {
        sogp.init(state_dim, output_dim, gk, noise, epsilon, capacity);

        //now we loop using a sine wave
        unsigned int max_itr = 100;
        VectorXd input(1);
        VectorXd output(1);

        VectorXd state;
        VectorXd prediction;
        VectorXd prediction_variance;

        VectorXd all_input = VectorXd::Random(max_itr)*2*M_PI;

        for (unsigned int i=0; i<max_itr; i++) {
            input(0) = all_input(i);//sin(i*0.01);
            output(0) = sin(input(0)*0.01);//sin((i+1)*0.01);

            //predict
            sogp.predict(input, prediction, prediction_variance);
            double error = (prediction - output).norm();
            cout << "Error: " << error << ", " << sogp.getCurrentSize() << endl;


            //train
            sogp.train(input, output);

        }
    } catch (OTLException &e) {
        e.showError();
    }

}

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

        std::cout << "Copy test" << std::endl;
        GaussianKernel gk2(gk);
        if (gk.eval(x,y) - 0.145876 > eps) {
            std::cout << "Gaussian Kernel copy produced a WRONG answer!!! " << std::endl;
        } else {
            std::cout << "Gaussian Kernel copy produced a RIGHT answer!!! " << std::endl;
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

int main(void) {
    try {
        storkgpValgrindTest();
    } catch (OTLException &e) {
        e.showError();
    }
    return 0;


//    try {
//        oesgpTest();
//    } catch (OTLException &e) {
//        e.showError();
//    }
//    return 0;

    try {
        sinTestSTORKGP();
    } catch (OTLException &e) {
        e.showError();
    }
    return 0;

    try {
        sinTestSOGPWin();
    } catch (OTLException &e) {
        e.showError();
    }
    return 0;


    SOGPMultidimTest();
    return 0;

    SOGPTest();
    return 0;

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

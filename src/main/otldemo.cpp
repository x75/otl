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
#include <iostream>

using namespace OTL;
using namespace std;

int main(int argc, char **argv) {
    //some code here
    Window delay_window;
    delay_window.init(1,1, 10);

    VectorXd features;

    for (unsigned int i=0; i<5; i++) {
        VectorXd input(1);
        input(0) = i;
        delay_window.update(input);
    }

    delay_window.getFeatures(features);
    std::cout << "Features: " << endl;
    cout << features << endl;

    //save and reload
    delay_window.save("test_window.data");

    Window test_window;
    test_window.load("test_window.data");
    VectorXd loaded_features;
    delay_window.getFeatures(loaded_features);

    std::cout << "Loaded Features: " << endl;
    cout << loaded_features << endl;

    return 0;
}

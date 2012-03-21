/**
  OTL Sliding Window Class.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements a sliding window for temporal regression

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_WINDOW_3281907590812038290178498638726509842309432
#define OTL_WINDOW_3281907590812038290178498638726509842309432

#include "otl_aug_state.h"
#include "otl_helpers.h"
#include <fstream>
#include <exception>
#include <string>

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;


/**
    OTL Window Class. The window class takes in a input and appends it to
    a sliding window. Note that newer inputs are placed in the FRONT
    of the window.
  */

class Window : public AugmentedState {
public:

    /**
        \brief updates the internal window with the input
        \param input the input
    **/
    virtual void update(const VectorXd &input);

    /**
        \brief Returns the size of the augmented state
    **/
    virtual unsigned int getStateSize(void);

    /**
        \brief Gets the internal window and puts it into the State parameter
        \param State where the window gets put into.
    **/
    virtual void getState(VectorXd &State);

    /**
        \brief Sets the internal window to the parameter State
        \param State a VectorXd with the State you want to set. Make sure it is the right size.
    **/
    virtual void setState(const VectorXd &State);

    /**
        \brief Resets the window to all zeros
    **/
    virtual void reset(void);

    /**
        \brief Saves the window State to a file
        \param filename a string specifying the filename to save to
    **/
    virtual void save(std::string filename);


    /**
        \brief Loads the window from a file
        \param filename a string specifying the filename to load from
    **/
    virtual void load(std::string filename);

    /**
        \brief Sets up the window.
        \param input_dim the input dimension
        \param output_dim the output dimension
        \param window_length how long is the delay window? This value must be
                strictly positive. A delay window of 1 means no windowing.
    **/
    void init(unsigned int input_dim, unsigned int output_dim, unsigned int window_length);

private:
    //Windowing parameters
    unsigned int input_dim;
    unsigned int output_dim;
    unsigned int window_length;
    unsigned int total_window_size;
    bool initialized;
    VectorXd window;
};



}

#endif

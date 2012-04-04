#include "otl_window.h"

namespace OTL {

void Window::update(const VectorXd &input) {
    VectorXd temp = this->window.segment(0,this->total_window_size - this->input_dim);
    window.segment(this->input_dim, this->total_window_size-this->input_dim) = temp;

    window.segment(0, this->input_dim) = input;
    return;
}


inline unsigned int Window::getStateSize(void) {
    return this->total_window_size;
}


void Window::getState(VectorXd &State) {
    State = this->window;
    return;
}


void Window::setState(const VectorXd &State) {
    if (State.rows() != this->total_window_size) {
        throw OTLException("Sorry, the feature you want to set is not the right length");
    }
    //set the State
    this->window = State;
    return;
}


void Window::reset(void) {
    this->window = VectorXd::Zero(this->total_window_size);
    return;
}

void Window::save(std::string filename) {

    std::ofstream out;
    try {
        out.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }

    out << this->input_dim << std::endl;
    out << this->output_dim << std::endl;
    out << this->window_length << std::endl;
    out << this->total_window_size << std::endl;
    out << this->initialized << std::endl;
    saveVectorToStream(out, this->window);
    out.close();
}

void Window::load(std::string filename) {
    std::ifstream in;
    try {
        in.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }

    in >> this->input_dim;
    in >> this->output_dim;
    in >> this->window_length;
    in >> this->total_window_size;
    in >> this->initialized;
    readVectorFromStream(in, this->window);
    in.close();
}


void Window::init(unsigned int input_dim, unsigned int output_dim, unsigned int window_length) {
    if (window_length == 0) {
        throw OTLException("You need a window bigger than 0. (Try 1 if don't want a window).");
    }
    if (input_dim == 0) {
        throw OTLException("Input dimension cannot be zero.");
    }
    if (output_dim == 0) {
        throw OTLException("Output dimension cannot be zero.");
    }

    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->window_length = window_length;
    this->total_window_size = input_dim*window_length;

    //initialise the window
    this->window = VectorXd::Zero(this->total_window_size);
    this->initialized = true;
    return;
}

}

#ifndef OTL_WINDOW_3281907590812038290178498638726509842309432
#define OTL_WINDOW_3281907590812038290178498638726509842309432

#include "otl_aug_feature.h"

namespace OTL {
class Window : public AugmentedFeatures {
public:
    virtual void update(const VectorXd &input) {
        return;
    }

    virtual void getFeatures(VectorXd &features) {
        return;
    }

    virtual void setFeatures(const VectorXd &features) {
        return;
    }

    virtual void reset(void) {
        return;
    }

    /**
        \brief Sets up the window.
        \param input_dim the input dimension
        \param output_dim the output dimension
        \param window_length how long is the delay window? This value must be
                strictly positive. A delay window of 1 means no windowing.
    **/
    void init(unsigned int window_length) {
        if (window_length == 0) {
            throw OTLException("You need a window bigger than 0. (Try 1 if don't want a window).");
        }

        this->window_length = window_length;
        return;
    }

private:
    //Windowing parameters
    unsigned int window_length;

};
}

#endif

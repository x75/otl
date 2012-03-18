#ifndef OTL_RESERVOIR_H_893201789576921893812908490127348470398
#define OTL_RESERVOIR_H_893201789576921893812908490127348470398

#include "otl_aug_state.h"

namespace OTL {

class Reservoir : public AugmentedState {
public:
    //types of activation functions
    //we only support one for now
    enum {
        TANH = 0
    };
    Reservoir(void) {
        return;
    }

    virtual void update(const VectorXd &input) {

    }

    virtual void getState(VectorXd &State) {

    }

    virtual void setState(const VectorXd &State) {

    }

    virtual void reset(void) {

    }

    virtual void save(std::string filename) {

    }

    virtual void load(std::string filename) {

    }

    /**
      \brief Sets up the reservoir
      \param input_dim the input dimension
      \param output_dim the output dimension
      \param reservoir_size how big is your reservoir?
      \param input_weight How much to weight your inputs (typically 1)
      \param output_feedback_weight Set this >0 if you want output feedback
                with the specified weighting
      \param activation function only OTLParams::TANH is supported for now.
      \param leak_rate the leak rate (between 0 and 1.0) of the reservoir
                (depends on your application)
      \param connectivity connectivity of the reservoir (between 0 and 1.0).
                Typically small e.g., 0.01 or 0.1
      \param spectral_radius the spectral radius of the reservoir. This should
                be < 1.0 but you can set it higher if you want.
      \param random_seed the random seed to initialise the reservoir. The same
                random seed will generate the same reservoir.
      **/
    void init(
            unsigned int input_dim,
            unsigned int output_dim,
            unsigned int reservoir_size,
            double input_weight,
            double output_feedback_weight,
            int activation_function,
            double leak_rate,
            double connectivity,
            double spectral_radius,
            unsigned int random_seed
            )
    {
        this->input_dim = input_dim;
        this->output_dim = output_dim;
        this->reservoir_size = reservoir_size;
        this->input_weight = input_weight;
        this->output_feedback_weight = output_feedback_weight;
        this->activation_function = activation_function;
        this->leak_rate = leak_rate;
        this->connectivity = connectivity;
        this->spectral_radius = spectral_radius;
        this->random_seed = random_seed;

        return;
    }

private:
    //Which augmented feature type?
    int augmented_feature_type;

    //common parameters
    unsigned int input_dim;
    unsigned int output_dim;

    //Reservoir parameters
    unsigned int reservoir_size;
    double input_weight;
    double output_feedback_weight;
    unsigned int activation_function;
    double leak_rate;
    double connectivity;
    double spectral_radius;
    unsigned int random_seed;

};

}
#endif

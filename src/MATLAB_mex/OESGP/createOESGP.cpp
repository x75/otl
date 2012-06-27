#include "mex.h" 

#include <iostream>
#include "otl.h"
#include "otl_oesgp.h"
#include "ObjectHandle.h"
using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	
    double* i_input_dim = mxGetPr(prhs[0]); 
    double* i_output_dim = mxGetPr(prhs[1]);
    double* i_reservoir_size = mxGetPr(prhs[2]);
    double* i_input_weight = mxGetPr(prhs[3]);
    double* i_output_feedback_weight = mxGetPr(prhs[4]);
    double* i_activation_function = mxGetPr(prhs[5]);
    double* i_leak_rate = mxGetPr(prhs[6]);
    double* i_connectivity = mxGetPr(prhs[7]);
    double* i_spectral_radius = mxGetPr(prhs[8]);
    double* i_use_inputs_in_state = mxGetPr(prhs[9]);
    double* i_kernel_parameters = mxGetPr(prhs[10]);
    double* i_noise = mxGetPr(prhs[11]);
    double* i_epsilon = mxGetPr(prhs[12]);
    double* i_capacity = mxGetPr(prhs[13]);
    double* i_random_seed = mxGetPr(prhs[14]);
    
	//assign the inputs
	unsigned int input_dim = (unsigned int) (i_input_dim[0]);
	unsigned int output_dim = (unsigned int) (i_output_dim[0]);
	
    unsigned int reservoir_size = (unsigned int) (i_reservoir_size[0]);
    double input_weight = i_input_weight[0];
    double output_feedback_weight = i_output_feedback_weight[0];
    int activation_function = (int) (i_activation_function[0]);
    double leak_rate = i_leak_rate[0];
    double connectivity = i_connectivity[0];
    double spectral_radius = i_spectral_radius[0];
    bool use_inputs_in_state = (bool) i_use_inputs_in_state[0];
    
    double noise = i_noise[0];
    double epsilon = i_epsilon[0];
    unsigned int capacity = i_capacity[0];
    int random_seed = i_random_seed[0];

    //assign the kernel parameters
	unsigned int m = mxGetM(prhs[10]);
	unsigned int n = mxGetN(prhs[10]);
	
	unsigned int len = (m > n) ? m : n;
	
	VectorXd kernel_parameters(len);
	for (unsigned int i=0; i<len; i++) {
		kernel_parameters[i] = i_kernel_parameters[i]; 
	}

	
	//initialise storkgp object
	OESGP *oesgp = new OESGP();
	try {
        oesgp->init( input_dim, output_dim, reservoir_size,
                    input_weight, output_feedback_weight,
                    activation_function,
                    leak_rate,
                    connectivity, spectral_radius,
                    use_inputs_in_state,
                    kernel_parameters,
                    noise, epsilon, capacity, random_seed);
        
	} catch (OTLException &e) {
		e.showError();
	}
 
	plhs[0] = create_handle(oesgp);
}
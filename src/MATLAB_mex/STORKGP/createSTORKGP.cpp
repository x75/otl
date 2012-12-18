#include "mex.h" 

#include "otl.h"
#include "otl_storkgp.h"
#include "ObjectHandle.h"
using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	
	double *i_input_dim;
	double *i_output_dim;
	double *i_tau;
	double *i_kernel_params;
	double *i_noise;
	double *i_epsilon;
	double *i_capacity;
	
	double l;
	double rho;
	double alpha;
	unsigned int input_dim;
	unsigned int output_dim;
	
	unsigned int tau;
	double noise;
	double epsilon;
	unsigned int capacity;
	
	i_input_dim =  mxGetPr(prhs[0]);
	i_output_dim = mxGetPr(prhs[1]);
	i_tau = mxGetPr(prhs[2]);
	i_kernel_params = mxGetPr(prhs[3]);
	i_noise = mxGetPr(prhs[4]);
	i_epsilon = mxGetPr(prhs[5]);	
	i_capacity = mxGetPr(prhs[6]);
	
	//assign the inputs
	input_dim = (unsigned int) i_input_dim[0];
	output_dim = (unsigned int) i_output_dim[0];
	tau = (unsigned int) i_tau[0];	
	noise = i_noise[0];
	epsilon = i_epsilon[0];
	capacity = (unsigned int) i_capacity[0];

    //assign the kernel parameters
	unsigned int m = mxGetM(prhs[3]);
	unsigned int n = mxGetN(prhs[3]);
	
	unsigned int len = (m > n) ? m : n;
	
	VectorXd kernel_parameters(len+1);
	for (unsigned int i=0; i<len; i++) {
		kernel_parameters[i] = i_kernel_params[i]; 
	}
	kernel_parameters[len] = (double) input_dim;

	
	//initialise storkgp object
	STORKGP *storkgp = new STORKGP();
	try {
		storkgp->init(input_dim, output_dim, tau, STORKGP::RECURSIVE_GAUSSIAN, kernel_parameters,
			noise, epsilon, capacity);
	} catch (OTLException &e) {
		e.showError();
	}
 
	plhs[0] = create_handle(storkgp);
}
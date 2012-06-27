#include "mex.h" 

#include <iostream>
#include "otl.h"
#include "otl_storkgp.h"
#include "ObjectHandle.h"

using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	double *i_storkgp_object;
    double *i_outputs;

	i_storkgp_object =  mxGetPr(prhs[0]);
	i_outputs = mxGetPr(prhs[1]);
    
    //assign the kernel parameters
	unsigned int m = mxGetM(prhs[1]);
	unsigned int n = mxGetN(prhs[1]);
	
	unsigned int len = (m > n) ? m : n;
    VectorXd outputs(len);
	for (unsigned int i=0; i<len; i++) {
		outputs[i] = i_outputs[i]; 
	}

    ObjectHandle<STORKGP>* handle = 
            ObjectHandle<STORKGP>::from_mex_handle(prhs[0]);
	handle->get_object().train(outputs);

}
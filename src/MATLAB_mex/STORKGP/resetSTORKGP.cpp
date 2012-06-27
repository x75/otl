#include "mex.h" 

#include <iostream>
#include "otl.h"
#include "otl_storkgp.h"
#include "ObjectHandle.h"

using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	double *i_storkgp_object;

	i_storkgp_object =  mxGetPr(prhs[0]);
    
    ObjectHandle<STORKGP>* handle = 
            ObjectHandle<STORKGP>::from_mex_handle(prhs[0]);
	handle->get_object().resetState();

}
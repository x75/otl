#include "mex.h" 

#include <iostream>
#include "otl.h"
#include "otl_oesgp.h"
#include "ObjectHandle.h"

using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	double *i_oesgp_object;

	i_oesgp_object =  mxGetPr(prhs[0]);
    
    ObjectHandle<OESGP>* handle = 
            ObjectHandle<OESGP>::from_mex_handle(prhs[0]);
	handle->get_object().resetState();

}
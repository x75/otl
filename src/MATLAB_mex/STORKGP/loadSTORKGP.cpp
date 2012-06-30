#include "mex.h" 

#include <iostream>
#include "otl.h"
#include "otl_storkgp.h"
#include "ObjectHandle.h"
#include <string>

using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	double *i_storkgp_object;
    char *i_filename;
    

    int buflen = (mxGetM(prhs[1]) * mxGetN(prhs[1]));
    
	i_storkgp_object = mxGetPr(prhs[0]);
	i_filename = mxArrayToString(prhs[1]);
    
    std::string filename(i_filename, buflen);
    
	ObjectHandle<STORKGP>* handle = 
	ObjectHandle<STORKGP>::from_mex_handle(prhs[0]);
	handle->get_object().load(filename);
    
    
    mxFree(i_filename);

}
#include "mex.h" 

#include <iostream>
#include "otl.h"
#include "otl_storkgp.h"
#include <string>

using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	double *i_storkgp_object;
    char *i_filename;
    

    int buflen = (mxGetM(prhs[1]) * mxGetN(prhs[1])) + 1;
    
	i_storkgp_object = mxGetPr(prhs[0]);
	i_filename = mxArrayToString(prhs[1]);
    
    std::string filename(i_filename, buflen);
    
    
    
    
    mxFree(i_filename);

}
#include "mex.h" 

#include <iostream>
#include "otl.h"
#include "otl_oesgp.h"
#include "ObjectHandle.h"

using namespace OTL;

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
	if (nlhs != 0 || nrhs != 1)
		mexErrMsgTxt("Usage: destorySTORKGP(h)");

	destroy_object<OESGP>(prhs[0]);
}
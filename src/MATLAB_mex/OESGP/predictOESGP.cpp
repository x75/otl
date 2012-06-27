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
    
    VectorXd prediction;
    VectorXd prediction_variance;
    
    ObjectHandle<OESGP>* handle = 
            ObjectHandle<OESGP>::from_mex_handle(prhs[0]);
	handle->get_object().predict(prediction, prediction_variance);
    
    
    if (nlhs >= 1) {
        plhs[0] = mxCreateDoubleMatrix(prediction.rows(), 1, mxREAL);
        double *output = mxGetPr(plhs[0]);
        for (unsigned int i=0; i<prediction.rows(); i++) {
            output[i] = prediction[i];
        }
    } 
    
    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(prediction_variance.rows(), 1, mxREAL);
        double *output = mxGetPr(plhs[1]);
        for (unsigned int i=0; i<prediction_variance.rows(); i++) {
            output[i] = prediction_variance[i];
        }        
    } 
}
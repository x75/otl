#include "mex.h" 
#include "math.h"
#include "stdio.h"

double sqNormdiff(double *x, double *y, double *scale, int length) {
    double normdiff = 0.0;
    int i;
    for (i=0; i<length; i++) {
        normdiff += pow((x[i] - y[i]),2.0); //scale[i]*(x[i] - y[i]),2);
    }
    return normdiff;
}

double getRecKerVal(double* x, double *y, double *scale, int m, int n,
        double sigma, double sigmai, double leak_rate)
{
    double k_val=0.0;
    int i;
    for (i=0; i<m; i++) {
        double d = sqNormdiff( &(x[i*n]), &(y[i*n]), scale, n );
        //kval = exp(- norm( ret_rate*(x(i,:) - y(i,:)) )^2/(2*sigmai^2))* ...
        //    exp((kval -1.0)/(sigma^2));
        k_val = exp( -d/(2*sigmai*sigmai) )*exp( (k_val - 1.0)/(sigma*sigma));
        
    }
    //printf("%lf\n", k_val);
    return k_val;
}
  

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
    //get the dimensions of the data
    int m, n;
    double *x;
    double *y;
    double *scale;
    double *params;
    double sigma, sigmai, leak_rate;
    double *output;
    double kernval;
    
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    
    x = mxGetPr(prhs[0]);
    y = mxGetPr(prhs[1]);
    params = mxGetPr(prhs[2]);

    sigmai = params[0];
    sigma = params[1];
    leak_rate = params[2];
    
    
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    output = mxGetPr(plhs[0]);
    kernval = getRecKerVal(x, y, scale, m, n,
         sigma,  sigmai, leak_rate);

    output[0] = kernval;
} 
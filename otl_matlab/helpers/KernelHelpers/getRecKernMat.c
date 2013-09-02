/* getRecKernMat - gets the recursive kernel
 *
 * Copyright (c) 2012, 2013 Harold Soh */

#include "mex.h"
#include <math.h>
//#include "sse_mathfun.h"
#include "fastonebigheader.h"

/*
 * #define DEBUG_MODE
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int tau;
    double *ell;
    double *a, *b, *C;
    double rho, rho2, ori_rho, z, prev_rke, r;
    double rke, drke, prev_drke, tro;
    int m,n, D, d;
    int m_ell, hyp;
    int i, j, k, t;
    int same_mat;
    int start_pos;
    
    same_mat = 0;
    
    if (nrhs < 5) {
        mexErrMsgTxt("Usage: getRecKernMat(ell, tau, rho, i, x, z)\n");
        return;
    }
    
    
    /*usage: getRecKernMat()*/
    ell = mxGetPr(prhs[0]);
    m_ell = mxGetM(prhs[0]);
    
    tau = (mxGetPr(prhs[1]))[0];
    rho = (mxGetPr(prhs[2]))[0];
    rho2 = rho*rho;
    /*ori_rho = -log(1/rho - 1);
     * tro = (1+exp(-ori_rho));
     * tro = z*z;*/
    
    /*the hyperparam to compute the derivatives*/
    hyp = (mxGetPr(prhs[3]))[0];
    
    a = mxGetPr(prhs[4]);
    m = mxGetN(prhs[4]);
    D = mxGetM(prhs[4]);
    
    if (nrhs == 6) {
        b = mxGetPr(prhs[5]);
        n = mxGetN(prhs[5]);
    } else {
        b = a;
        n = m;
        same_mat = 1;
    }
    
    
#ifdef DEBUG_MODE
    printf("D: %d \n", D);
    for (i=0; i<m_ell; i++) {
        printf("ell %d: %lf \n", i, ell[i]);
    }
    printf("tau: %d \n", tau);
    printf("rho: %f \n", rho);
    printf("m: %d \n", m);
    printf("n: %d \n", n);
    printf("d: %d \n", d);
#endif
    
    if (D%tau != 0) {
        mexErrMsgTxt("Input dimension is not correct. Are you sure the tau is correct?");
        return;
    }
    
    
    
    d = D/tau;
    if (m_ell != d) {
        mexErrMsgTxt("Dimension of ell does not match dimension of input");
        return;
    }
    
    /*create our outputs*/
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    C = mxGetPr(plhs[0]);
    
    if (hyp == 0) {
        /*perform regular recursive kernel computation*/
        for (i=0; i<m; i++) {
            
            start_pos = 0;
            if (same_mat) {
                start_pos = i;
            }
            
            for (j=start_pos; j<n; j++) {
                
                prev_rke = 1.0;
                
                for (t=0; t<tau; t++) {
                    /*first compute the square difference*/
                    z = 0.0;
                    for (k=0; k<d; k++) {
                        
                        r = (a[D*i+ t*d +k] - b[D*j+ t*d +k])/ell[k];
                        z += r*r;
                        
#ifdef DEBUG_MODE
                        printf("------------------\n");
                        printf("a_%d : %lf \n", t, a[D*i+ t*d +k]);
                        printf("b_%d : %lf \n", t, b[D*j+ t*d +k]);
                        printf("r_%d : %lf \n", t, r);
                        printf("z_%d : %lf \n", t, z);
                        printf("------------------\n");
#endif
                        
                    }
                    
                    /*compute recursive kernel*/
                    rke = exp(-z/2 + (prev_rke - 1) / rho);
                    prev_rke = rke;
                    
#ifdef DEBUG_MODE
                    printf("------------------\n");
                    printf("rke : %lf \n", rke);
                    printf("------------------\n");
#endif
                    
                }
                
                C[i+j*m] = rke;
                if (same_mat) {
                    C[j+i*m] = rke;
                }
            }
        }
    } else if (hyp <= d) {
        /*we are computing derivatives for the characteristic lengthscales*/
        for (i=0; i<m; i++) {
            
            start_pos = 0;
            if (same_mat) {
                start_pos = i;
            }
            
            
            for (j=start_pos; j<n; j++) {
                
                prev_rke = 1.0;
                prev_drke = 0.0;
                for (t=0; t<tau; t++) {
                    /*first compute the square difference*/
                    z = 0.0;
                    for (k=0; k<d; k++) {
                        
                        r = (a[D*i+ t*d +k] - b[D*j+ t*d +k])/ell[k];
                        z += r*r;
                        
#ifdef DEBUG_MODE
                        printf("------------------\n");
                        printf("a_%d : %lf \n", t, a[D*i+ t*d +k]);
                        printf("b_%d : %lf \n", t, b[D*j+ t*d +k]);
                        printf("r_%d : %lf \n", t, r);
                        printf("z_%d : %lf \n", t, z);
                        printf("------------------\n");
#endif
                        
                    }
                    
                    /*compute recursive kernel*/
                    rke = exp(-z/2 +  (prev_rke - 1) / rho);
                    prev_rke = rke;
                    
                    /*compute derivative for this hyper param*/
                    k = hyp-1;
                    r = (a[D*i+ t*d + k] - b[D*j+ t*d +k])/ell[k];
                    z = r*r;
                    
                    drke = rke * (prev_drke/rho + z);
                    prev_drke = drke;
                    
#ifdef DEBUG_MODE
                    printf("------------------\n");
                    printf("rke : %lf \n", rke);
                    printf("------------------\n");
#endif
                    
                }
                
                C[i+j*m] = drke;
                if (same_mat) {
                    C[j+i*m] = drke;
                }
                
                
            }
        }
    } else if (hyp == d + 2) {
        for (i=0; i<m; i++) {
            
            start_pos = 0;
            if (same_mat) {
                start_pos = i;
            }
            
            
            for (j=0; j<n; j++) {
                prev_rke = 1.0;
                prev_drke = 0.0;
                for (t=0; t<tau; t++) {
                    /*first compute the square difference*/
                    z = 0.0;
                    for (k=0; k<d; k++) {
                        
                        r = (a[D*i+ t*d +k] - b[D*j+ t*d +k])/ell[k];
                        z += r*r;
                    }
                    
                    /*compute recursive kernel*/
                    rke = exp(-z/2 + (prev_rke - 1) / rho);
                    
                    
                    /*compute derivative for this hyper param*/
                    if (t>0) {
                        drke = rke * (prev_drke/rho + (1- prev_rke)/rho2);
                        prev_drke = drke;
                    }
                    prev_rke = rke;
#ifdef DEBUG_MODE
                    printf("------------------\n");
                    printf("rke : %lf \n", rke);
                    printf("------------------\n");
#endif
                    
                }
                
                
                C[i+j*m] = drke;/* (-1*exp(-ori_rho)/ tro);*/
                /*C[i+j*m] = drke;*/
                if (same_mat) {
                    C[j+i*m] = drke;
                }
                
            }
        }
    } else {
        printf("Hyp = %d ", hyp);
        mexErrMsgTxt("but No such hyperparameter for this kernel.\n");
    }
}
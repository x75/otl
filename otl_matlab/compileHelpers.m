cd helpers/KernelHelpers
disp 'Compiling sq_dist.c'
mex sq_dist.c
disp 'Compiling getRecKernMat.c'
mex getRecKernMat.c
cd ../
disp 'Done!'
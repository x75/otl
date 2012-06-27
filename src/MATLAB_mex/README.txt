MATLAB Experimental Bindings

In this folder, you'll find experimental bindings for STORKGP and OESGP. 
Within each respective directory, there is a compile[ALGNAME]_mex.m file 
while compiles the relevant mex bindings. You *may* have to edit this file
to point to the relevant eigen path.

Although you can use the mex functions (e.g., createSTORKGP) directly, 
it's often much easier to use the classes. For example, for STORKGP, 
take a look at the commented testSTORKGP.m file which shows you how to 
use the storkgp class. A similar class (and example file) exists for 
OESGP. 

If you have questions, send me an email at haroldsoh@imperial.ac.uk.

Have fun!


Online Temporal Learning (OTL) Library
Copyright Harold Soh 2011, 2012
haroldsoh@gmail.com
haroldsoh@imperial.ac.uk

== Introduction ==
The Online Temporal Learning (OTL) C++ library implements several memory structures (a neural reservoir and sliding window) and online learning methods (RLS, Sparse Online Gaussian Process). Included in the library are two "high-level" online learning methods: the Online Echo State Gaussian Process (OESGP) and the Spatial-Temporal Online Recursive Kernel Gaussian Process (STORK-GP). Academic papers detailing these methods:

OESGP: H. Soh and Y. Demiris, “Iterative Temporal Learning and Prediction with the Sparse Online Echo State Gaussian Process”, International Joint Conference on Neural Networks (IJCNN-2012), Brisbane, Australia (to appear) 

STORKGP: H. Soh and Y. Demiris, "Iteratively Learning to Classify Objects by Touch with Online Spatio-Temporal Experts", Under Review. Contact me for a pre-print. 

OTL currently in pre-alpha; it works but documentation is still sorely lacking. Check out the Getting Started Guides for using the high-level methods (OESGP and STORKGP). Example code is within the src/examples directory. I've tried to make it easy to do the standard thing, i.e., learn from and predict sequential time-series data. If that's what you're trying to accomplish, it should be relatively easy to get things going. 

That said, at its current state, OTL is meant for use by individuals familiar with C++/Python programming on a *nix environment. Feel free to give it a shot and I'll be happy to answer questions via email. However, please note that depending my workload, answers may be *very* slow in coming. 

Have fun! For more up-to-date information, refer to the OTL wiki: https://bitbucket.org/haroldsoh/otl/wiki/Home

== Pre-requisites ==
OTL requires the Eigen C++ Matrix library and cmake. It also needs SWIG for the Python bindings. You can download Eigen at http://eigen.tuxfamily.org. 

If you're using Ubuntu, you can install these pre-reqs using apt-get:

sudo apt-get install libeigen3-dev cmake swig

or using the Software Center.

== Installation ==
Clone the repository using:

hg clone ssh://hg@bitbucket.org/haroldsoh/otl

Change into the otl directory, create a build directory and cmake as usual.

cd otl
mkdir build
cd build
cmake ../
make

Everything should compile and you should have a libOTL.a library file. 

== Options: Python Bindings and Doxygen Documentation ==
If you want the Python bindings and the Doxygen generated documentation, please turn on the appropriate option:

cmake ../ -DBUILD_PYTHON_BINDINGS=ON -DBUILD_DOCS=ON

If you specified the Python bindings, you'll have additional .so files and Doxygen comments will be in the doc folder.

=== Experimental Matlab Bindings ===
Experimental MATLAB bindings are available for the STORKGP algorithm. OESGP bindings will soon be available. This has been tested on a 64-bit machine.

To use these bindings, head to the src/MATLAB_mex directory.
Fire up MATLAB and edit the compileSTORKGP_mex.m file. Run it. 
You should then be able to use the mex files. 

== Getting Started Guides ==
For the getting started guides, please refer to the OTL Wiki: https://bitbucket.org/haroldsoh/otl/wiki/Home

== Usage and License ==
This code is DUAL-LICENSED. 

If you are using this code in non-commercial settings, e.g.,  at academic institutions, teaching institutes or for hobby, learning and research: you can use this code freely, i.e., make copies, distribute and modify the code for any non-commercial purpose provided the original license and this copyright notice are included. If you do use this code in your research, please cite:

@inproceedings{Soh2012,
Author = {Harold Soh and Yiannis Demiris},
Booktitle = {International Joint Conference on Neural Networks (IJCNN)},
Title = {Iterative Temporal Learning and Prediction with the Sparse
Online Echo State Gaussian Process},
Year = {2012}}

or 

@inproceedings{Soh2012,
Author = {Harold Soh and Yiannis Demiris},
Booktitle = {Under Review},
Title = {Iteratively Learning to Classify Objects by Touch with Online Spatio-Temporal Experts},
Year = {2012}}

(whichever is more appropriate). 

If you use this code for commercial use: You must contact the author for a commercial license; if you make financial gain out of this library, I would appreciate a fair share to be able to continue to work on this code, support my family and the causes I believe in.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



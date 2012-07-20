%module otl_storkgp
 %{
    /* Includes the header in the wrapper code */
    #include "otl_storkgp.h"
    using namespace OTL;
    using Eigen::VectorXd;
    using Eigen::MatrixXd;
 %}

%include "std_string.i"

namespace Eigen {
//convert a list to a Eigen VectorXd
%typemap(in) VectorXd& {

	int length = PyList_Size($input);
	if (length < 0) {
		length = 1;
	}

	VectorXd *temp = new VectorXd(length);

    for (int i=0; i<length; i++) {
		(*temp)[i] = PyFloat_AsDouble(PyList_GetItem($input, i));
	}
	$1 = temp;

}

//convert a list to a Eigen MatrixXd
%typemap(in) MatrixXd& {

	int nrows = PyList_Size($input);
	PyObject *list = PyList_GetItem($input, 0);
	int ncols = PyList_Size(list);

	if (nrows < 0) {
		nrows = 1;
	}
	if (ncols < 0) {
		ncols = 1;
	}

    //printf("%d %d\n", nrows, ncols);

	MatrixXd *temp = new MatrixXd(nrows, ncols);
    for (int i=0; i<nrows; i++) {
		PyObject *list = PyList_GetItem($input, i);
        for (int j=0; j<ncols; j++) {
			(*temp)(i,j) = PyFloat_AsDouble(PyList_GetItem(list, j));
		}
	}
	$1 = temp;
	//delete list;
}

//converts from a Eigen VectorXd to a list
%typemap(argout) VectorXd& {
    int length = (*$1).rows();
    int input_length = PyList_Size($input);

    if (length < 0) {
        length = 1;
    }

    PyList_SetSlice($input, 0, input_length, NULL );
    for (int i=0; i<length; i++) {
        //PyObject *val = SWIG_NewPointerObj(PyFloat_FromDouble((*$1)[i]), SWIGtype, 1);
        PyObject *val = PyFloat_FromDouble((*$1)[i]);
        PyList_Append($input, val);
        Py_DECREF(val);
    }
}

//converts from a Eigen MatrixXd to a list
%typemap(argout) MatrixXd& {
	int nrows = (*$1).rows();
	int ncols = (*$1).cols();
	printf("%d %d\n", nrows, ncols);

	int input_length = PyList_Size($input);

	if (input_length < 0) {
		input_length = 1;
	}

	PyList_SetSlice($input, 0, input_length, NULL );

    for (int i=0; i<nrows; i++) {
		PyObject *list = PyList_New(ncols);
        for (int j=0; j<ncols; j++) {
            PyObject *val = PyFloat_FromDouble((*$1)(i,j));
            PyList_SetItem(list, j, val);
            Py_DECREF(val);
		}
		PyList_Append($input, list);
        Py_DECREF(list);
	}
}


%typemap(freearg) VectorXd& {
   delete $1;
}

%typemap(freearg) MatrixXd& {
   delete $1;
}

}


class STORKGP {
public:
    /**
        types of kernels
       **/
    enum {
        RECURSIVE_GAUSSIAN = 0,
        RECURSIVE_EQUALITY_GAUSSIAN = 1
    };

    /**
      \brief Sets up the reservoir
      \param input_dim the input dimension
      \param output_dim the output dimension
      \param tau window size for approximation
      \param kernel_type the type of kernel (enum)
      \param kernel_parameters kernel parameters for the recursive kernel
      \param noise noise for SOGP
      \param epsilon threshold for SOGP
      \param capacity the capacity for the SOGP
      **/
    void init(
            unsigned int input_dim,
            unsigned int output_dim,
            unsigned int tau,
            int kernel_type,
            VectorXd &kernel_parameters,
            double noise,
            double epsilon,
            unsigned int capacity
            );
    /**
      \brief updates STORKGP window with input
      \param input the input (a VectorXd)
      */
    virtual void update(const VectorXd &input);

    /**
      \brief trains the STORKGP given the current state and the output
      \param output (a VectorXd)
      */
    virtual void train(const VectorXd &output);

    /**
      \brief make a prediction and prediction variance.
      \param prediction the output prediction (a VectorXd)
      \param prediction_variance the output prediction variance (a VectorXd)
      */
    virtual void predict(VectorXd &prediction, VectorXd &prediction_variance);

    /**
      \brief resets the state of the reservoir
      */
    virtual void resetState();

    /**
      \brief resets the SOGP algorthm
      */
    virtual void resetSOGP();

    /**
      \brief gets the state of the reservoir
      */
    virtual void getState(VectorXd &state);

    /**
      \brief sets the state of the reservoir
      */
    virtual void setState(const VectorXd &state);

    /**
      \brief resets the reservoir
      */
    virtual void reset(void);

    /**
        \brief Saves the reservoir to a file
        \param filename a string specifying the filename to save to
    **/
    virtual void save(std::string filename);

    /**
        \brief loads the model from disk
        \param filename where to load from?
    **/
    virtual void load(std::string filename);

    /**
      \brief returns the reservoir size
      **/
    virtual unsigned int getStateSize(void);

    /**
      \brief gets the current number of basis vectors
      \returns unsigned int (the number of basis vectors)
      */
    unsigned int getCurrentSize(void);

};




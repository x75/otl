#!/usr/bin/env python
__author__ = 'haroldsoh'

from otl_oesgp import OESGP
from numpy import sin, array
from numpy.linalg import norm

def SimpleTest():
    #Create our OESGP object
    oesgp = OESGP()

    #Our parameters
    dim = 1

    res_size = 100
    input_weight = 1.0
    output_feedback_weight = 0.1
    activation_function = 1
    leak_rate = 0.0
    connectivity = 0.1
    spectral_radius = 0.9
    kernel_params = [1.0, 1.0]
    noise = 0.01
    epsilon = 1e-3
    capacity = 100
    random_seed = 0

    #Initialise our OESGP
    oesgp.init(dim, dim, res_size, input_weight, output_feedback_weight,
        activation_function, leak_rate, connectivity, spectral_radius,
        False, kernel_params, noise, epsilon, capacity, random_seed)

    #loop through some sample code
    for i in range(0,1000):
        input = [sin(i*0.01)]
        output = [sin((i + 1)*0.01)]

        #update the reservoir
        oesgp.update(input)
        #oesgp.update_wfeedback(input, input)

        prediction = []
        variance = []
        #make a prediction
        oesgp.predict(prediction, variance)

        #get and print the error
        error = norm(array(prediction) - array(output))
        print "Error: ", error

        #train the model
        oesgp.train(output)

    #save the model for the future
    oesgp.save("test")

if __name__ == "__main__":
    SimpleTest()

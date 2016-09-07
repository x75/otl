#!/usr/bin/env python
__author__ = 'haroldsoh'

from otl_oesgp import OESGP
import numpy as np
from numpy import sin, array
from numpy.linalg import norm
import matplotlib.pylab as pl

def plot_state(res_size, r_t, input_t, output_t, prediction_t, variance_t):
    pl.subplot(311)
    pl.title("res activation")
    pl.gca().clear()
    sel = slice(0, res_size, 10)
    print sel
    pl.plot(r_t[:,sel])
    pl.subplot(312)
    pl.gca().clear()
    pl.plot(input_t)
    pl.subplot(313)
    pl.gca().clear()
    pl.plot(output_t)
    pl.plot(prediction_t)
    pl.plot(prediction_t + variance_t, "r-", lw=0.5)
    pl.plot(prediction_t - variance_t, "r-", lw=0.5)
    pl.draw()
    

def SimpleTest():
    # Create our OESGP object
    oesgp = OESGP()

    # Our parameters
    dim = 1

    res_size = 100
    input_weight = 1.0
    output_feedback_weight = 0.1
    activation_function = 0
    leak_rate = 0.0
    connectivity = 0.1
    spectral_radius = 0.9
    kernel_params = [1.0, 1.0]
    noise = 0.1
    epsilon = 1e-3
    capacity = 100
    random_seed = 100

    # experiment boundaries
    len_episode = 1000
    input_t      = np.zeros((len_episode, dim))
    output_t     = np.zeros((len_episode, dim))
    prediction_t = np.zeros((len_episode, dim))
    variance_t = np.zeros((len_episode, dim))

    r = [] # np.zeros((res_size, 1))
    r_t = np.zeros((len_episode, res_size))
    
    # Initialise our OESGP
    oesgp.init(dim, dim, res_size, input_weight, output_feedback_weight,
        activation_function, leak_rate, connectivity, spectral_radius,
        False, kernel_params, noise, epsilon, capacity, random_seed)

    pl.ion()
    #loop through some sample code
    for i in range(0,len_episode):
        input = [sin(i*0.01)]
        output = [sin((i + 1)*0.01)]

        # print input
        
        input_t[i,0] = input[0]
        output_t[i,0] = output[0]

        #update the reservoir
        oesgp.update(input)
        #oesgp.update_wfeedback(input, input)

        # get state
        oesgp.getState(r)
        r_a = np.array(r)
        r_a += np.random.normal(0., 0.01, r_a.shape)
        oesgp.setState(r_a.tolist())
        # print r
        r_t[i,:] = r_a
        
        prediction = []
        variance = []
        #make a prediction
        oesgp.predict(prediction, variance)

        # print prediction
        # print variance
        prediction_t[i,0] = prediction[0]
        variance_t[i,0]   = variance[0]

        #get and print the error
        error = norm(array(prediction) - array(output))
        print "Error: ", error

        #train the model
        oesgp.train(output)

        if i % 100 == 0:
            plot_state(res_size, r_t, input_t, output_t, prediction_t, variance_t)
            
    #save the model for the future
    oesgp.save("test")

    pl.ioff()
    plot_state(res_size, r_t, input_t, output_t, prediction_t, variance_t)
    pl.show()

if __name__ == "__main__":
    SimpleTest()

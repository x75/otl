#!/usr/bin/env python
__author__ = 'haroldsoh, Oswald Berthold'

import sys
from otl_oesgp import OESGP
from otl_storkgp import STORKGP
import numpy as np
from numpy import sin, array
from numpy.linalg import norm
import pylab as pl

if "/home/src/QK/smp/lib" not in sys.path:
    sys.path.insert(0, "/home/src/QK/smp/lib")
from smp.reservoirs import Reservoir


class SimpleTest(object):
    def __init__(self, mode=0):
        # Create our OESGP object
        self.oesgp = OESGP()
        self.oesgp10 = OESGP()
        self.storkgp = STORKGP()

        # Our parameters
        self.dim = 1

        self.res_size = 100 # 20
        self.input_weight = 2.0
        self.output_feedback_weight = 0.3
        self.activation_function = 0
        self.leak_rate = 0.1
        self.connectivity = 0.1
        self.spectral_radius = 1.5
        
        self.kernel_params = [1.0, 1.0]
        self.noise = 0.1
        self.epsilon = 1e-3
        self.capacity = 100
        self.random_seed = 100

        # experiment boundaries
        self.len_episode = 50000
        self.input_t      = np.zeros((self.len_episode, self.dim))
        self.output_t     = np.zeros((self.len_episode, self.dim))
        self.prediction_t = np.zeros((self.len_episode, self.dim))
        self.variance_t = np.zeros((self.len_episode, self.dim))
        self.output10_t     = np.zeros((self.len_episode, self.dim))
        self.prediction10_t = np.zeros((self.len_episode, self.dim))
        self.variance10_t = np.zeros((self.len_episode, self.dim))

        self.r = [] # np.zeros((res_size, 1))
        self.r_t = np.zeros((self.len_episode, self.res_size))

        # custom extensions
        self.res_theta = 0.1
        self.res_theta_state = 0.01
        self.coeff_a = 0.2
    
        # Initialise our OESGP
        self.oesgp.init(self.dim, self.dim, self.res_size, self.input_weight,
                    self.output_feedback_weight, self.activation_function,
                    self.leak_rate, self.connectivity, self.spectral_radius,
                    False, self.kernel_params, self.noise, self.epsilon,
                    self.capacity, self.random_seed)
        self.oesgp10.init(self.dim, self.dim, self.res_size, self.input_weight,
                    self.output_feedback_weight, self.activation_function,
                    self.leak_rate, self.connectivity, self.spectral_radius,
                    False, self.kernel_params, self.noise, self.epsilon,
                    self.capacity, self.random_seed)

        self.storkgp.init(self.dim, self.dim,
                          self.res_size, # window size
                          0, # kernel type
                          [0.5, 0.99, 1.0, self.dim],
                          1e-4,
                          1e-4,
                          100
                          )

        self.res = Reservoir(N=self.res_size,
                             p = self.connectivity,
                             input_num=self.dim,
                             output_num=self.dim,
                             g = self.spectral_radius,
                             tau = (1. - self.leak_rate),
                             eta_init = 0,
                             feedback_scale = 0, #self.output_feedback_weight,
                             input_scale = self.input_weight,
                             bias_scale = 0,
                             nonlin_func = np.tanh, # lambda x: x,
                             sparse = True, ip=False,
                             theta = self.res_theta,
                             theta_state = self.res_theta_state,
                             coeff_a = self.coeff_a
                             )
        # self.res.init_wi_random(mu = 0., std=0.2)
        # print "wi", self.res.wi
        # self.res.init_wi_ones()
        # print "wi", self.res.wi


    def run(self):
        pl.ion()
        # pl.plot([0, 1])
        # pl.draw()
        # loop through some sample code
        for i in range(0,self.len_episode):
            input = [sin(i*0.01)]
            output = [sin((i + 1)*0.01)]
            output10 = [sin((i + 10)*0.01)]
            # input = [sin(i*0.01)+sin(i*0.02)]
            # output = [sin((i + 1)*0.01)+sin((i + 1)*0.02)]

            print "type(input)", type(input), input
        
            self.input_t[i,0] = input[0]
            self.output_t[i,0] = output[0]
            self.output10_t[i,0] = output10[0]

            # update the OTL reservoir, this is much faster than going through smpres code
            self.oesgp.update(input)
            self.oesgp10.update(input)
            # self.storkgp.update(input)
            # oesgp.update_wfeedback(input, input)

            # # update my reservoir
            # self.res.execute(input)
            # # print self.res.r.flatten().tolist()
            # self.oesgp.setState(self.res.r.flatten().tolist())
            # self.oesgp10.setState(self.res.r.flatten().tolist())
            # # self.storkgp.setState(self.res.r.flatten().tolist())
            # # print self.res.r.shape
            # # print self.res.r.tolist()

            # get state
            self.oesgp.getState(self.r)
            # self.storkgp.getState(self.r)
            # print self.r
            r_a = np.array(self.r)
            # r_a += np.random.normal(0., 0.01, r_a.shape)
            # self.oesgp.setState(r_a.tolist())
            # self.r_t[i,:] = r
            self.r_t[i,:] = r_a
            # self.r_t[i,:] = self.res.r.reshape((self.res_size))
        
            prediction = []
            variance = []
            prediction10 = []
            variance10 = []
            #make a prediction
            self.oesgp.predict(prediction, variance)
            self.oesgp10.predict(prediction10, variance10)
            # self.storkgp.predict(prediction, variance)

            # print prediction
            # print variance
            self.prediction_t[i,0] = prediction[0]
            self.variance_t[i,0]   = variance[0]
            self.prediction10_t[i,0] = prediction10[0]
            self.variance10_t[i,0]   = variance10[0]

            # get and print the error
            error = norm(array(prediction) - array(output))

            # train the model
            print "type(output)", type(output), output
            self.oesgp.train(output)
            self.oesgp10.train(output10)
            # self.storkgp.train(output)

            if i % 100 == 0:
                print "plotting state"
                self.plot_state()
                print "Error[%d] = %f" % (i, error)
            
        # save the model for the future
        self.oesgp.save("test")

        pl.ioff()
        self.plot_state()
        pl.show()


        # freerunning
        inputs = []
        predictions = []
        prediction = []
        variances = []
        variance = []
        inputs.append(output[-1])
        self.oesgp.update([output[-1]])
        self.oesgp.predict(prediction, variance)
        predictions.append(prediction[0])
        variances.append(variance[0])
        # print prediction, variance
        for i in range(1000):
            # sample new input from prev output
            new_in = np.random.normal(prediction[0], variance[0])
            # new_in = prediction[0]
            inputs.append(new_in)
            # self.oesgp.update([predictions[-1]])
            self.oesgp.update([new_in])
            prediction = []
            variance = []
            self.oesgp.predict(prediction, variance)
            print i, new_in, prediction[0], variance[0]
            predictions.append(prediction[0])
            variances.append(variance[0])
        predictions = np.asarray(predictions)
        variances = np.asarray(variances)
        pl.subplot(211)
        pl.plot(inputs)
        pl.subplot(212)
        pl.plot(predictions, "k-")
        pl.plot(predictions + variances, "r-")
        pl.plot(predictions - variances, "r-")
        pl.show()
        
        
    def plot_state(self):
        pl.subplot(411)
        pl.gca().clear()
        pl.title("res activation")
        sel = slice(0, self.res_size, 10)
        # print sel
        pl.plot(self.r_t[:,sel])
        # 
        pl.subplot(412)
        pl.gca().clear()
        pl.title("res input")
        pl.plot(self.input_t)
        pl.subplot(413)
        pl.gca().clear()
        pl.title("res output,pred,var")
        pl.plot(self.output_t)
        pl.plot(self.prediction_t)
        pl.plot(self.prediction_t + self.variance_t, "r-", lw=0.5)
        pl.plot(self.prediction_t - self.variance_t, "r-", lw=0.5)
        pl.subplot(414)
        pl.gca().clear()
        pl.title("res10 output,pred,var")
        pl.plot(self.output10_t)
        pl.plot(self.prediction10_t)
        pl.plot(self.prediction10_t + self.variance10_t, "r-", lw=0.5)
        pl.plot(self.prediction10_t - self.variance10_t, "r-", lw=0.5)
        pl.draw()
    
        

if __name__ == "__main__":
    # modes: default soesgp, storkgp, myresoesgp
    st = SimpleTest()
    st.run()

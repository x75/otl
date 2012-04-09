#!/usr/bin/env python
__author__ = 'haroldsoh'

from otl_storkgp import STORKGP
from numpy import sin, array
from numpy.linalg import norm

def SimpleTest():
    storkgp = STORKGP()
    dim = 100
    storkgp.init(dim, dim, 5, STORKGP.RECURSIVE_GAUSSIAN,  [0.1, 0.99, 1.0, 1], 0.001, 1e-3, 100)

    for i in range(0,100):
        input = [sin(i*0.01)]*dim
        output = [sin((i + 1)*0.01)]*dim
        storkgp.update(input)
        prediction = []
        variance = []
        storkgp.predict(prediction, variance)
        error = norm(array(prediction) - array(output))
        #print error

        storkgp.train(output)

if __name__ == "__main__":
    for rep in range(100):
        print rep
        SimpleTest()

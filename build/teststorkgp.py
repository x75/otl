#!/usr/bin/env python
__author__ = 'haroldsoh'

from otl_storkgp import STORKGP
from numpy import sin, array
from numpy.linalg import norm

storkgp = STORKGP()
storkgp.init(1, 1, 10, [0.1, 0.99, 1.0, 1], 0.001, 1e-3, 100)

for i in range(0,1000):
    input = [sin(i*0.01)]
    output = [sin((i + 1)*0.01)]
    storkgp.update(input)
    prediction = []
    variance = []
    storkgp.predict(prediction, variance)
    error = norm(array(prediction) - array(output))
    print error

    storkgp.train(output)


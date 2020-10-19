# -*- coding: utf-8 -*-
"""
Created on Wed June 5 14:25:20 2019

@author: Lenovo
"""
import numpy as np
import random

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

a = 1 #learning rate
bias = 1 #value of bias
weights_1 = [[0.8,0.2],[0.4, 0.9]] #weights generated in a list (3 weights in total for 2 neurons and the bias)
weights_2 = [[0.3, 0.5], [0.7, 0.2]]
weights_output = [0.6,0.3]
#weightsoutput = [random.random(),random.random()]
weight = random.random()
y = 1


def Neuron(input1, input2, layer, weights, neuron):

    outputP = input1 * weights[layer][neuron] + input2 * weights[layer][neuron+1]
    out = sigmoid(outputP)
    return [out, outputP]

def lastNeuron(input1, input2, layer, weights, neuron):
    outputP = input1 * weights[neuron] + input2 * weights[neuron+1]

    out = sigmoid(outputP)
 #   print("hidden: ", neuron)
 #   print(out)
    return [out, outputP]

for i in range(400):

    #feedforward
    [neuron1, output1] = Neuron(1,1,0,weights_1,0)
    [neuron2, output2] = Neuron(1,1,1,weights_1,0)
    [neuron3, output3] = Neuron(neuron1,neuron2,0,weights_2,0)
    [neuron4, output4] = Neuron(neuron1, neuron2,1,weights_2,0)
    [neuron5, output] = lastNeuron(neuron3,neuron4, 2,weights_output,0)
    
    results = [neuron3, neuron4]
    results2 = [output3, output4]

    #backpropagation
    error = y - output
    delta = error * sigmoid_derivative(output)
    z5_delta = delta * sigmoid_derivative(neuron5)
    z4_delta = delta * sigmoid_derivative(neuron4)
    z3_delta = delta * sigmoid_derivative(neuron3)
    z2_delta = delta * sigmoid_derivative(neuron2)
    z1_delta = delta * sigmoid_derivative(neuron1)
    weights_1[0] += z1_delta
    weights_1[1] += z2_delta
    weights_2[0] += z3_delta
    weights_2[1] += z4_delta
    weights_output[0] += z5_delta
  #  print(weights_1)
  #  print(weights_2)
print("Predicted output: ", output)

# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:32:59 2019

@author: Lenovo
"""

import random
a = 1 #learning rate
bias = 1 #value of bias
weights = [random.random(),random.random(),random.random(),random.random(),random.random(),random.random()] #weights generated in a list (3 weights in total for 2 neurons and the bias)
print(weights)

def Perceptron(input1, input2, output) :
   outputP = input1*weights[0]+input2*weights[1]+bias*weights[2]
   if outputP > 0 : #activation function (here Heaviside)
      outputP = 1
   else :
      outputP = 0
   error = output - outputP
   weights[0] += error * input1 * a
   weights[1] += error * input2 * a
   weights[2] += error * bias * a
   return weights
   
for i in range(50) :
   weights = Perceptron(1,1,1) #True or true
   print(weights)
   weights = Perceptron(1,0,1) #True or false
   weights = Perceptron(0,1,1) #False or true
   weights = Perceptron(0,0,0) #False or false
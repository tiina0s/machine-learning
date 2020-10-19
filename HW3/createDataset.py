# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:18:26 2019

@author: Lenovo
"""
import random
import numpy as np

dataset = []
for i in range(500):
    x = random.random()
    x2 = random.random()
    x3 = random.random()
    x4 = random.random()
    #if x > 0.5 and x2 > 0.5:
    #    y = 1
    #else:
    #    y = 0
    #if x > 0.2 and x < 0.24:
    #    y = 0
    #if x2 < 0.1 and x2 > 0.07:
    #    y = 1
    y = np.random.randint(2)
    set1 = [x,x2,x3,x4,y]
    dataset.append(set1)
    
print(dataset)

f= open("data.txt","w")

for i in range(500):
     f.write(str(dataset[i]))
     f.write('\n')
     
f.close()
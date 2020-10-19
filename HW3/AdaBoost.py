# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:23:26 2019

@author: Lenovo
"""

def train(X,y,test_size):
    print("training")
    
f=open("data.txt", "r")
contents =f.readlines()
#print(contents)
data = []
X = []
y = []
for line in contents:
    data.append(line.rstrip('\n'))
    x_i = line.split(',')
    xx = x_i[0].replace('[','')
   # print(xx)
    x_n = [x_i[0].replace('[',''),x_i[1],x_i[2],x_i[3]]
    
    X.append(x_n)
    y.append(x_i[4].rstrip('\n]'))
  #  print(x_n)
    
train_data = data[70:]
test_data = data[:30]

print(len(test_data))
print(len(train_data))

train_x = []
train_y = []
test_x = []
test_y = []


for i in test_data:
    data_test = i.split(',')
    test_x.append([data_test[0].replace('[',''),data_test[1],data_test[2],data_test[3]])
    test_y.append([data_test[4].rstrip(']').replace(' ','')])
for i in train_data:
    data_train = i.split(',')
    train_x.append([data_train[0],data_train[1],data_train[2],data_train[3]])
    train_y.append([data_train[4].rstrip(']').replace(' ','')])
  
print(test_y)
#print(data)
#X_train, X_test, y_train, y_test = train(X, y, test_size=0.3)


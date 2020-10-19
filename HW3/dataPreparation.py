# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:23:26 2019

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scipy.stats as sps
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def train(x,y):
    print("training")
    
f=open("data.txt", "r")
contents =f.readlines()
#print(contents)
data = []
X = []
y = []
for line in contents:
    data.append(line.rstrip('\n'))

    
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
    x1 = data_test[0].replace('[','')
    test_x.append([float(x1),float(data_test[1]),float(data_test[2]),float(data_test[3])])
    test_y.append([float(data_test[4].rstrip(']').replace(' ',''))])
for i in train_data:
    data_train = i.split(',')
    x2 = data_train[0].replace('[','')
    train_x.append([float(x2),float(data_train[1]),float(data_train[2]),float(data_train[3])])
    train_y.append([float(data_train[4].rstrip(']').replace(' ',''))])
  
print(test_y)
#print(data)
#train(train_x, train_y)

y = []
z = []
n = []
weight = []
n2 = len(train_x)
for i in train_x:
    y.append(i[0])
    z.append(i[1])
    weight.append(1/n2)
for i in train_y:
    n.append(i[0])
#y = train_x[0]
#z = train_x[1]
#n = train_y

#print(y)
#print(z)
#print(weight)
print("MNN")
print(n)

models = []
alphas = [] 
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
Evaluation = pd.DataFrame(train_y.copy())
#print(Evaluation['predictions'])
Evaluation['weights'] = weight

for i in range(1):
    model = clf.fit(train_x,train_y, sample_weight=np.array(Evaluation['weights']))
    models.append(model)
    y_pred = model.predict(train_x)
    
    #y_pred2 = clf.predict(train_x)
    print("Accuracy:",metrics.accuracy_score(train_y, y_pred))
    #print("Accuracy:",metrics.accuracy_score(train_y, y_pred2))
    
    error = 0
    for i in range(len(y_pred)):
      #  print(y_pred2[i])
     #   print(train_y[i])
        if (y_pred[i] != train_y[i]):
            error += 1
    print("ERREOR")
    print(error)
    print("pred")
    print(y_pred)
    Evaluation['predictions'] = y_pred
    Evaluation['target'] = n
 #   print("rped")
 #   print(Evaluation['predictions'])
    Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation['target'],1,0)
    print("eval")
    print(Evaluation['evaluation'])
    Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation['target'],1,0)
    # Calculate the misclassification rate and accuracy
    accuracy = sum(Evaluation['evaluation'])/len(Evaluation['evaluation'])
    misclassification = sum(Evaluation['misclassified'])/len(Evaluation['misclassified'])
    # Caclulate the error
    err = np.sum(Evaluation['weights']*Evaluation['misclassified'])/np.sum(Evaluation['weights'])
     
    print("erre")
    print(err)
    # Calculate the alpha values
    alpha = np.log((1-err)/err)
    print("alpha")
    alphas.append(alpha)
    print(alpha)
 #   print(Evaluation['misclassified'])
    Evaluation['weights'] *= np.exp(alpha*Evaluation['misclassified'])
    print("WEITH")
    print(Evaluation['weights'])
#fig, ax = plt.subplots()
#ax.scatter(z, y)

#for i, txt in enumerate(n):
   # ax.annotate(txt, (z[i], y[i]))
 #   hold on
#ax.scatter(train_x[0][:], train_x[1][:])
 
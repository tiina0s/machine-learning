# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:11:19 2019

@author: Lenovo
## https://www.python-course.eu/Boosting.php
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('data.csv',header=None)
dataset = dataset.sample(frac=1)
dataset.columns = ['x1','x2','x3','x4','target']

for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])
    

class Boosting:
    def __init__(self,dataset,T,test_dataset):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None
        
    
    def fit(self):

        X = self.dataset.drop(['target'],axis=1)
        Y = self.dataset['target'].where(self.dataset['target']==1,-1)
        
        Evaluation = pd.DataFrame(Y.copy())
        Evaluation['weights'] = 1/len(self.dataset) 
        
        
        alphas = [] 
        # List of all models
        models = []
        
        for t in range(self.T):

            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1) #            

            model = Tree_model.fit(X,Y,sample_weight=np.array(Evaluation['weights'])) 
            

            models.append(model)
            predictions = model.predict(X)
          #  print("Accuracy:",metrics.accuracy_score(train_y, y_pred))
            Evaluation['predictions'] = predictions
          #  print(predictions)
            Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation['target'],1,0)
            Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation['target'],1,0)

            err = np.sum(Evaluation['weights']*Evaluation['misclassified'])/np.sum(Evaluation['weights'])
 
   
            alpha = np.log((1-err)/err)
            alphas.append(alpha)

            Evaluation['weights'] *= np.exp(alpha*Evaluation['misclassified'])

 
        # mida suurem weight, seda valem tulemus
        print("Weights after each classifier: ")
        print(Evaluation['weights'])
        self.alphas = alphas
        self.models = models
        
    # Each models predictions
    def predict(self):
        X_test = self.test_dataset.drop(['target'],axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset['target'].reindex(range(len(self.test_dataset))).where(self.dataset['target']==1,0)
    
        
        accuracy = []
        predictions = []
        
        for alpha,model in zip(self.alphas,self.models):
            prediction = alpha*model.predict(X_test) 
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]))

        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))
   
f=open("data.txt", "r")
contents =f.readlines()

data = []
X = []
y = []
for line in contents:
    data.append(line.rstrip('\n'))

    
train_data = data[70:]
test_data = data[:30]

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
for i in train_y:
    n.append(i[0])
        
        
number_of_base_learners = 5
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
for i in range(number_of_base_learners):
    model = Boosting(dataset,i,dataset)
    model.fit()
    model.predict()
ax0.plot(range(len(model.accuracy)),model.accuracy,'-b')
ax0.set_xlabel('weak models count ')
ax0.set_ylabel('accuracy')
print('Nr of models: ',number_of_base_learners,' accuracy: ',model.accuracy[-1]*100,'%')    
                 
plt.show() 
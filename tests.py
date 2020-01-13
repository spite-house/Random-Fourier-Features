# -*- coding: utf-8 -*-
"""
Spyder Editor

This is the main testing file (where things are meant to be run from)
@Authors: Kevin Mah
"""

#Test of algorithm 1 [Random Fourier Features] from 'Random Features for Large Scale Kernel Machines'

import csv
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt

with open('adult.csv', 'r') as f:
    reader = csv.reader(f)
    raw_examples = list(reader)


'''
the list of all traits recorded with keys to represent the values they take
'''



def make_key():
    key_list = list()
    key_list.append("continuous")
    key_list.append(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    key_list.append("continuous")
    key_list.append(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
    key_list.append("continuous")
    key_list.append(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    key_list.append(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    key_list.append(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    key_list.append(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    key_list.append(["Female", "Male"])
    key_list.append("continuous")
    key_list.append("continuous")
    key_list.append("continuous")
    key_list.append(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])
    return key_list

def rescale_data(data):
    processed = list()
    targets = list()
    for i in range(len(data)):
        if not int(data[i][20]) > 100:
            processed.append(data[i][1:23])
            targets.append(data[i][23])
    processed = np.array(processed)
    targets = np.array(targets)
    features_list = list()
    for i in range(len(processed[0])):
         vector = cont_trait_vector(processed,i)
         feature = np.array([vector])
         features_list.append(feature.T)
    features = np.concatenate(features_list, axis = 1)
    return features, targets
    

def split_cases(x, y):
    neg = list()
    pos = list()
    for i in range(len(x)):
        if y[i] == -1:
            neg.append(x[i])
        else:
            pos.append(x[i])
    return pos, neg

def remove_incomplete(rawdata):
    complete = list()
    m = len(rawdata[0])
#    print(rawdata[0])
    for example in rawdata:
        a = 0
        for i in range(m):
            if example[i] == '?':
                a = 1
        if a == 0:
            complete.append(example)
#    print(len(complete))
#    print(len(rawdata))
    return complete

def process_data(data, key_list):
    features_list = list()
    m = len(key_list[0])
    for i in range(len(key_list)):
        vector = trait_vector(data, i)
        if key_list[i] == "continuous":
            vector = cont_trait_vector(data,i)
            feature = np.array([vector])
            features_list.append(feature.T)
        else:
            features_list.append(make_rep(vector, key_list[i]))
    features = np.concatenate(features_list, axis = 1)
    return features

#i is entry in data list
def trait_vector(data, trait):
    vector = list()
    for i in range(len(data)):
        vector.append(data[i][trait])
    return vector

def cont_trait_vector(data, trait):
    vector = list()
    for i in range(len(data)):
        vector.append(data[i][trait])
    rescale = float(max(vector))
    for i in range(len(vector)):
        vector[i] = float(vector[i])/rescale
    return vector

# '<=50k' = -1
# '>50k' = 1
def make_target(data):
    entry = len(data[0]) - 1
    vector = list()
    for i in range(len(data)):
        vector.append(int((-1)**(data[i][entry][0] == '<')))
    y = np.array([vector]).T
    return y

#consider n classes
#data m x n matrix
def make_rep(labels, key):
    n = len(key)
    rep = np.zeros([len(labels), n - 1])
    for i in range(len(labels)):
        for j in range(n-1):
            if labels[i] == key[j]:
                rep[i, j] = 1
    return rep


raw_examples = raw_examples[1:len(raw_examples)]
data = remove_incomplete(raw_examples)
key_list = make_key()
y = make_target(data)
x = process_data(data, key_list)


#x, y = rescale_data(raw_examples)

n = len(x)
splitindex = .7
x_train = x[0:int(splitindex*n)].T
y_train = y[0:int(splitindex*n)].T
x_test = x[int(splitindex*n):n].T
y_test = y[int(splitindex*n):n].T

x_train = x_train.astype(float)
y_train = y_train.astype(float)
x_test = x_test.astype(float)
y_test = y_test.astype(float)


xmtrain = matlab.double(x_train.tolist())
ymtrain = matlab.double(y_train.tolist())
xmtest = matlab.double(x_test.tolist())
ymtest = matlab.double(y_test.tolist())

eng = matlab.engine.start_matlab()
#perf = eng.regressiontest(xmtrain, ymtrain, xmtest, ymtest, 'gaussian', 'rp_factorize_large_real', 0, 500, nargout = 3)

variables = list()
errors = list()
times = list()

#d tests

for d in [10, 100, 250, 300, 500, 1000, 1500, 2000]:
    perf = eng.regressiontest(xmtrain, ymtrain, xmtest, ymtest, 'gaussian', 'rp_factorize_large_real', 1, d, nargout = 3)
    variables.append(d)
    errors.append(perf[1])
    times.append(perf[2])
    plt.plot(variables, errors)
    print('d = ' + str(d) + '  ' + 'Error: ' + str(perf[1]) + '   ' + 'Training time: ' + str(perf[2]))

'''
for l in [0, .01, .1, 1, 10, 100]:
    perf = eng.regressiontest(xmtrain, ymtrain, xmtest, ymtest, 'gaussian', 'rp_factorize_large_real', l, 10, nargout = 3)
    variables.append(l)
    errors.append(perf[1])
    times.append(perf[2])
    plt.plot(variables, errors)
    print('lambda = ' + str(l) + '  ' + 'Error: ' + str(perf[1]) + '   ' + 'Training time: ' + str(perf[2]))
'''
plt.plot(variables, errors, 'ro')
plt.xscale('log')
plt.show()
#perf = eng.regressiontest(xmtrain, ymtrain, xmtest, ymtest, 'laplacian', 'rpbin', .1, 50, nargout = 3)
#print(perf[1])
#print(process_data(data))
#trait = trait_vector(data, 9)
#trait = trait[0:20]
#rep = make_rep(trait,["Female", "Male"])
#print(rep)

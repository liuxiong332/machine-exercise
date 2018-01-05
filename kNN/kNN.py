from numpy import array, tile, sum, empty
import operator
from collections import defaultdict
import re

def createDataSet():
    group = array([
        [1, 1.1], [1, 1], [0, 0], [0, 1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset, labels, k):
    row_count = dataset.shape[0]
    inset = tile(inX, (row_count, 1))
    deltaset = dataset - inset
    set_indices = sum(deltaset ** 2, axis=1).argsort()[0:k]
    
    label_counts = defaultdict(int)
    for indice in set_indices:
        label = labels[indice]
        label_counts[label] += 1
    return sorted(label_counts.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def file2matrix(filename):
    with open(filename) as fr:
        readlines = fr.readlines()
    ret_mat = empty((len(readlines), 3))
    labels = []
    for index, line in enumerate(readlines):
        texts = line.strip().split('\t')
        ret_mat[index, :] = texts[0:3]
        labels.append(texts[-1])
    return ret_mat, labels

def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    deltavals = maxvals - minvals
    norm_dataset = dataset - tile(minvals, [dataset.shape[0], 1])
    norm_dataset = norm_dataset / deltavals
    return norm_dataset, minvals, deltavals 

def datingClassTest():
    hotrate = 0.1
    dating_matrix, labels = file2matrix('./datingTestSet.txt')
    dating_matrix, *s = autonorm(dating_matrix)
    testrow_count = int(len(labels) * hotrate)
    error_count = 0
    for row_i in range(0, testrow_count):
        res_label = classify0(dating_matrix[row_i, :], dating_matrix[testrow_count:], labels[testrow_count:], 5)
        if res_label != labels[row_i]:
            error_count += 1
    print('error percentage:', error_count /  float(testrow_count))

def classifyPerson():
    percent_tat = float(input("percentage of time spent playing vide games"))
    ff_miles = float(input('Frequent filter miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    data_mat, labels = file2matrix('./datingTestSet.txt')
    norm_dataset, minvals, deltavals = autonorm(data_mat)
    inarr = array([ff_miles, percent_tat, ice_cream])
    inarr = (inarr - minvals) / deltavals
    label = classify0(inarr, norm_dataset, labels, 3)
    print('You will probably like this persion: ', label)
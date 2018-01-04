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
        labels.push(texts[-1])
    return ret_mat, labels

group, labels = createDataSet()
print(classify0([0, 0], group, labels, 3))
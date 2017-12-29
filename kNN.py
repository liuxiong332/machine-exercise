from numpy import array, tile, sum
import operator
from collections import defaultdict

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
    sorted(label_counts, key=operator)

import kNN 
import os
import numpy as np

def img2vector(filename):
    vector = np.empty(32 * 32)

    with open(filename) as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        vector[i * 32:i * 32 + 32] = list(map(lambda x: int(x), line.strip()))
    return vector

def handwriting_test():
    training_path = os.path.join(os.path.dirname(__file__), 'digits', 'trainingDigits')
    filenames = os.listdir(training_path)
    vec_matrix = np.empty((len(filenames), 32 * 32))
    labels = []
    for i, filename in enumerate(filenames):
        abspath = os.path.join(training_path, filename)
        vec_matrix[i] = img2vector(abspath)
        labels.append(filename.split('_')[0])

    test_path = os.path.join(os.path.dirname(__file__), 'digits', 'testDigits')
    testfiles = os.listdir(test_path)
    test_count = len(testfiles)
    error_count = 0
    for filename in testfiles:
        abspath = os.path.join(test_path, filename)
        vector = img2vector(abspath)
        result = kNN.classify0(vector, vec_matrix, labels, 3)
        label = filename.split('_')[0]
        if result != label:
            print('The real result is %s, but the calc result is %s' % (label, result))            
            error_count += 1
    print('The total error rate is %f' % (error_count / float(test_count)))
    
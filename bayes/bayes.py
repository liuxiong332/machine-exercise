import numpy as np

def load_dataset():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I',  'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec

def get_vocab_list(dataset):
    vabaset = set()
    for doc in dataset:
        vabaset = vabaset | set(doc)
    return list(vabaset)

def words2vec(vocab_list, input_set):
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] += 1
        else:
            print('The word: %s is not in my vocabulary' % word)
    return ret_vec

def train_nb(train_matrix, train_category):
    train_count = len(train_matrix)
    word_count = len(train_matrix[0])
    p_abusive = sum(train_category) / train_count  # p(1)
    p0_nums = np.ones(word_count)
    p1_nums = np.ones(word_count)
    p0_denom, p1_denom = 2, 2
    for i in range(train_count):
        if train_category[i]:
            p1_nums += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_nums += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vec = p1_nums / p1_denom
    p0_vec = p0_nums / p0_denom
    return p_abusive, p0_vec, p1_vec

def classify_nb(vec, pclass1, p0_vec, p1_vec):
    p0 = np.log(1 - pclass1) + np.dot(vec, np.log(p0_vec))
    p1 = np.log(pclass1) + np.dot(vec, np.log(p1_vec))
    return 1 if p1 > p0 else 0

def testing_nb():
    list_posts, list_classes = load_dataset()
    vocab_list = get_vocab_list(list_posts)
    train_matrix = []
    for postin_doc in list_posts:
        train_matrix.append(words2vec(vocab_list, postin_doc))
    p_abusive, p0_vec, p1_vec = train_nb(train_matrix, list_classes)

    test_entry = ['love', 'my', 'dalmation']
    test_vec = words2vec(vocab_list, test_entry)
    print(test_entry, 'classified as: ', classify_nb(test_vec, p_abusive, p0_vec, p1_vec))

    test_entry = ['stupid', 'garbage']
    test_vec = words2vec(vocab_list, test_entry)
    print(test_entry, 'classified as: ', classify_nb(test_vec, p_abusive, p0_vec, p1_vec))
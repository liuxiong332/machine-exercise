from collections import defaultdict
from math import log, inf
import operator

# 计算香农熵
def calc_shannon_ent(dataset):
    label_counts = defaultdict(int)
    for rowdata in dataset:
        label_counts[rowdata[-1]] += 1 
    ent_val = 0
    entries_count = len(dataset)
    for key, count in label_counts.items():
        pk = count / entries_count
        ent_val -= pk * log(pk, 2)
    return ent_val

# 返回 特征值索引axis == value的数据集（剔除axis列）
def split_dataset(dataset, axis):
    valdata_map = defaultdict(list)
    for row in dataset:
        val = row[axis]
        valdata_map[val].append(row[:axis] + row[axis + 1:])
    return valdata_map

def calc_subent(dataset, axis):
    valdata_map = split_dataset(dataset, axis)
    ent_sum = 0
    entries_count = len(dataset)
    for val, items in valdata_map.items():
        prob = len(items) / entries_count
        ent_sum == prob * calc_shannon_ent(items)
    return ent_sum

def choose_bestfeature(dataset):
    feature_num = len(dataset[0]) - 1
    min_ent = inf
    for i in range(feature_num):
        subent = calc_subent(dataset, i)
        if subent < min_ent:
            min_ent = subent
            best_axis = i
    return best_axis

def max_label(dataset):
    counts = defaultdict(int)
    for row in dataset:
        counts[row[-1]] += 1
    return max(counts.items(), key=operator.itemgetter(1))[0]

def create_tree(dataset, labels):
    first_val = dataset[0][-1]
    # 所有数据都属于一个类别
    if all(row[-1] == first_val for row in dataset):
        return first_val
    elif len(dataset[0]) == 1: # 没有特征属性可以进行决策
        return max_label(dataset)   
    axis = choose_bestfeature(dataset) #选取最合适的索引
    feature = labels[axis]      # 根据索引获取feature名字
    feature_tree = {}
    val_map = split_dataset(dataset, axis)
    for val, subset in val_map.items():
        new_labels = labels[0:axis] + labels[axis + 1:]
        feature_tree[val] = create_tree(subset, new_labels)
    return { feature: feature_tree }

def classify(input_tree, feat_labels, test_vec):
    label = list(input_tree.keys())[0]
    index = feat_labels.index(label)
    test_val = test_vec[index]
    second_tree = input_tree[label].get(test_val)
    if isinstance(second_tree, dict):
        return classify(second_tree, feat_labels, test_vec)
    return second_tree

def retrieve_tree(i):
    list_trees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'yes'}}}}
    ]
    return list_trees[i]

def get_tree_depth(feature_tree):
    depth = 1
    feature_axe = list(feature_tree.keys())[0]
    second_dict = feature_tree[feature_axe]
    for val in second_dict.values():
        if isinstance(val, dict):
            depth = max(depth, get_tree_depth(val) + 1)
        else:
            depth = max(depth, 1)
    return depth

def get_leafs_count(feature_tree):
    leaf_count = 0
    axe = list(feature_tree.keys())[0]
    second_dict = feature_tree[axe]
    for val in second_dict.values():
        if isinstance(val, dict):
            leaf_count += get_leafs_count(val)
        else:
            leaf_count += 1
    return leaf_count


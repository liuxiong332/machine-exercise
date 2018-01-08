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

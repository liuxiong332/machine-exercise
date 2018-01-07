from collections import defaultdict
from math import log, inf

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
    

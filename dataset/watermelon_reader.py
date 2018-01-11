import re

def process_data(row_data):
    row_data[-2] = float(row_data[-2])
    row_data[-3] = float(row_data[-3])
    del row_data[0]
    return row_data

def read_data():
    with open('watermelons.txt', encoding='utf-8') as fr:
        labels = fr.readline().strip().split()[1:]
        dataset = [process_data(line.strip().split()) for line in fr.readlines()]
    return labels, dataset
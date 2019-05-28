import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from math import log


class Node:
    def __init__(self, root=None, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self,val,node):
        self.tree[val]=node


class DecisionTree:
    def __init__(self):

    # 经验熵
    @staticmethod
    def calc_emp_entropy(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 0
        emp_entropy = -sum([(p / data_length) * log(p / data_length) for p in label_count.values()])
        return emp_entropy

    # 经验条件熵
    def calc_cond_entropy(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_entropy = sum([(len(p) / data_length) * self.calc_emp_entropy(p) for p in feature_sets.values()])
        return cond_entropy

    # 信息增益
    @staticmethod
    def calc_info_gain(entropy, cond_entroy):
        return entropy - cond_entroy

    def info_gain_train(self, datasets):
        feature_count = len(datasets[0]) - 1
        info_gain = []
        emp_entropy = self.calc_emp_entropy(datasets)
        for i in range(feature_count):
            c_info_gain = self.calc_info_gain(emp_entropy, self.calc_cond_entropy(datasets, axis=i))
            info_gain.append((i, c_info_gain))
        choice_feature = max(info_gain, key=lambda x: x[-1])
        return choice_feature

    def train(self, train_data):
        train_x, train_y = load_iris(return_X_y=True)
        if len(train_y.value_counts()) == 1:
            return Node

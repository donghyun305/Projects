import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from functools import partial
from random import random, seed

class Generate_Data(object):
    def __init__(self, data_size):
        self.data_size = data_size

    def original_data(self):
        seed(1)
        x = [2 * random() - 1 for i in range(self.data_size)]
        y = [2 * random() -1 for i in range(self.data_size)]
        class_label = [1 if 0.5 * i + j > 0 else -1 for i, j in zip(x, y)]
        return list(zip(class_label, x, y))

    def random_data(self, dim, ws=(0.5, 0.5)):
        seed(1)
        data = []
        n1, n2 = int(self.data_size * ws[0]), int(self.data_size * ws[1])
        for i in range(n1):
            x = [2 * random() - 1 for i in range(dim)]
            data.append((-1, *x))

        for j in range(n2):
            x = [2 * random() - 1 for j in range(dim)]
            data.append((1, *x))
        return data, n1, n2

    def split_data(self, data):
        train_set = []
        test_set = []
        for i in data:
            if random() < 0.8:
                train_set.append(i)
            else:
                test_set.append(i)
        return train_set, test_set

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from random import random, seed, randrange
from math import sqrt
import numpy as np
from functools import partial
from sklearn.model_selection import GridSearchCV

class Generate_Data(object):
    def __init__(self, data_size):
        self.data_size = data_size

    def original_data(self):
        seed(1)
        x = [2 * random() - 1 for _ in range(self.data_size)]
        y = [2 * random() - 1 for _ in range(self.data_size)]
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

    def add_noise(self, data, po1, po2):
        corrupted_data = []
        for i in data:
            if i[0] == -1:
                if random() < po1:
                    corrupted_data.append((1, i[1], i[2]))
                else:
                    corrupted_data.append(i)
            else:
                if random() < po2:
                    corrupted_data.append((-1, i[1], i[2]))
                else:
                    corrupted_data.append(i)
        return corrupted_data

    def split_data(self, data):
        train_set = []
        test_set = []
        for i in data:
            if random() < 0.8:
                train_set.append(i)
            else:
                test_set.append(i)
        return train_set, test_set

class TrainingModel(object):
    def __init__(self, data_size, is_random, po1, po2, dim=3, ws = [0.5, 0.5]):
        self.data_maker = Generate_Data(data_size)
        self.true_data_map = {}
        if is_random:
            noise_free_data, self.n1, self.n2 = self.data_maker.random_data(dim, ws)
            self.set_random = True
        else:
            noise_free_data = self.data_maker.original_data()
            self.n1 = self.n2 = self.data_maker.data_size // 2
            self.set_random = False
        self.init_true_data_map(noise_free_data)
        noised_data = self.data_maker.add_noise(noise_free_data, po1, po2)
        self.noised_train_set, self.noised_test_set = self.data_maker.split_data(noised_data)
        self.nosiy_test_map = {(x, y): label for label, x, y in self.noised_test_set}
        self.clf = None

    def init_true_data_map(self, noise_free_data):
        for label, x, y in noise_free_data:
            self.true_data_map[(x, y)] = label

    def train_svm(self, kernel='linear', C=1.0):
        X = [(x, y) for _, x, y in self.noised_train_set]
        y = [label for label, _, _ in self.noised_train_set]
        self.clf = svm.SVC(kernel=kernel, C=C)
        self.clf.fit(X, y)

    def test_accuracy(self):
        X_test = [(x, y) for _, x, y in self.noised_test_set]
        y_true = [label for label, _, _ in self.noised_test_set]
        y_pred = self.clf.predict(X_test)
        accuracy = sum([1 if y_t == y_p else 0 for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)
        return accuracy

    def cross_val_score(self, k, kernel='linear', C=1.0):
        kf = KFold(n_splits=k)
        X = [(x, y) for _, x, y in self.noised_train_set]
        y = [label for label, _, _ in self.noised_train_set]
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            clf = svm.SVC(kernel=kernel, C=C)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = sum([1 if y_t == y_p else 0 for y_t, y_p in zip(y_test, y_pred)]) / len(y_test)
            scores.append(accuracy)
        return np.mean(scores)

    def plot_decision_boundary(self):
        h = 0.02
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        X1 = [x for label, x, y in self.noised_test_set if label == -1]
        Y1 = [y for label, x, y in self.noised_test_set if label == -1]
        X2 = [x for label, x, y in self.noised_test_set if label == 1]
        Y2 = [y for label, x, y in self.noised_test_set if label == 1]
        plt.scatter(X1, Y1, c='blue', marker='s', edgecolors='k', label='Class -1')
        plt.scatter(X2, Y2, c='red', marker='o', edgecolors='k', label='Class 1')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

def unbiased_estimator_loss(l, t, y, rho_p, rho_n):
    return ((1 - rho_n) * l(t, y) - rho_p * l(t, -y)) / (1 - rho_p - rho_n)

def alpha_weighted_loss(l, t, y, alpha):
    return (1 - alpha) * l(t, y) + alpha * l(t, -y)


def train_model(X_train, y_train, rho_plus, rho_minus, alpha):
    unbiased_l = partial(unbiased_estimator_loss, rho_plus=rho_plus, rho_minus=rho_minus)
    alpha_weighted_l = partial(alpha_weighted_loss, alpha=alpha)

    parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svc = SVC()
    grid_search = GridSearchCV(svc, parameters, scoring='neg_log_loss')

    min_loss = float('inf')
    best_model = None

    for f in grid_search:
        model = SVC(C=f.C, kernel=f.kernel)
        model.fit(X_train, y_train)
        y_pred = model.decision_function(X_train)

        loss = 0
        for i in range(len(y_train)):
            t = y_pred[i]
            y = y_train[i]
            loss += alpha_weighted_l(unbiased_l, t, y)
        loss /= len(y_train)

        if loss < min_loss:
            min_loss = loss
            best_model = model

    return best_model

def loss_function(t, y, rho_p, rho_n):
    return ((1 - rho_n) * l(t, y) - rho_p * l(t, y))/ (1 - rho_p - rho_n)

def l(t, y):
    return max(0, 1 - t * y)

def empirical_risk(f, X, Y, rho_p, rho_n):
    n = len(Y)
    return np.mean([loss_function(f(X[i]), Y[i], rho_p, rho_n) for i in range(n)])

def noisy_train_model(X, Y, rho_p, rho_n, F):
    best_f = None
    min_risk = np.inf
    for f in F:
        risk = empirical_risk(f, X, Y, rho_p, rho_n)
        if risk < min_risk:
            min_risk = risk
            best_f = f
    return best_f

def svm_function_factory(clf):
    def f(x):
        return clf.predict([x])[0]
    return f

if __name__ == "__main__":
    data_size = 5000
    rho_p = 0.1
    rho_n = 0.1
    dim = 2
    ws = (0.5, 0.5)

    # Create a training model
    model = TrainingModel(data_size, is_random=False, po1=rho_p, po2=rho_n, dim=dim, ws=ws)

    # Train the SVM classifier
    model.train_svm(kernel='linear', C=1.0)

    # Test the accuracy of the trained model
    accuracy = model.test_accuracy()
    # Perform cross-validation
    k = 5
    cv_score = model.cross_val_score(k, kernel='linear', C=1.0)

    X = [(x, y)for _, x, y in model.noised_train_set]
    Y = [label for label, _, _ in model.noised_test_set]
    F = [svm_function_factory(model.clf)]
    noisy_train_model = noisy_train_model(X, Y, rho_p, rho_n, F)


    model.train_svm(kernel='linear', C=1.0)
    noisy_accuracy = model.test_accuracy()
    print("Noisy data test accuracy: {:.2f}%".format(noisy_accuracy * 100))

    # 노이즈가 있는 데이터에 대한 교차 검증 수행
    noisy_cv_score = model.cross_val_score(k, kernel='linear', C=1.0)
    print("Noisy data cross-validation score ({}-fold): {:.2f}%".format(k, noisy_cv_score * 100))


    # Plot the decision boundary
    clean_train_set, clean_test_set = model.data_maker.split_data([(label, x, y) for (x, y), label in model.true_data_map.items()])
    clean_model = TrainingModel(data_size, is_random=False, po1=0, po2=0, dim=dim, ws=ws)
    clean_model.noised_train_set = clean_train_set
    clean_model.noised_test_set = clean_test_set
    clean_model.train_svm(kernel='linear', C=1.0)
    clean_accuracy = clean_model.test_accuracy()
    print("Clean data test accuracy: {:.2f}%".format(clean_accuracy * 100))

    clean_cv_score = clean_model.cross_val_score(k, kernel='linear', C=1.0)
    print("Clean data cross-validation score ({}-fold): {:.2f}%".format(k, clean_cv_score * 100))



    # 결정 경계를 그립니다.
    print("Clean data decision boundary:")
    clean_model.plot_decision_boundary()

    print("Noisy data decision boundary:")
    model.plot_decision_boundary()


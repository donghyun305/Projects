import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_openml
import random

class Generate_Data(object):
    def __init__(self, noise_prob):
        self.noise_prob = noise_prob

    def add_noise(self, data, noise_prob=0.05):
        X_noisy = np.array([x for _, x in data])
        y = np.array([label for label, _ in data])

        for i, x_noisy in enumerate(X_noisy):
            for j in range(x_noisy.shape[0]):
                if random.random() < noise_prob:
                    X_noisy[i, j] = random.random()
        return X_noisy, y

class TrainingModel(object):
    def __init__(self, noise_prob):
        self.data_maker = Generate_Data(noise_prob)
        self.clf = svm.SVC(kernel='rbf', C=1.0)

        # MNIST 데이터 불러오기 및 전처리
        mnist = fetch_openml('mnist_784')
        X, y = mnist.data, mnist.target
        X = X.astype(np.float32) / 255.0
        self.X_train, self.X_test = X[:60000], X[60000:]
        self.y_train, self.y_test = y[:60000], y[60000:]

    def train_svm(self):
        X_train, y_train = self.data_maker.add_noise(list(zip(self.X_train, self.y_train)))
        self.clf.fit(X_train, y_train)

    def test_accuracy(self):
        y_true = self.y_test
        y_pred = self.clf.predict(self.X_test)
        accuracy = sum([1 if y_t == y_p else 0 for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)
        return accuracy

    def cross_val_score(self, k, kernel='rbf', C=1.0):
        kf = KFold(n_splits=k)
        X, y = self.X_train, self.y_train
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = svm.SVC(kernel=kernel, C=C)
            clf.fit(self.data_maker.add_noise(list(zip(X_train, y_train))))
            y_pred = clf.predict(X_test)
            accuracy = sum([1 if y_t == y_p else 0 for y_t, y_p in zip(y_test, y_pred)]) / len(y_test)
            scores.append(accuracy)
        return np.mean(scores)

if __name__ == '__main__':
    noise_prob = 0.1
    model = TrainingModel(noise_prob)

    # 깨끗한 데이터 정확도 및 교차 검증 정확도
    model.train_svm()
    acc_clean = model.test_accuracy()
    acc_cv_clean = model.cross_val_score(5)

    # 노이즈 있는 데이터 정확도 및 교차 검증 정확도
    model.train_svm()
    acc_noisy = model.test_accuracy()
    acc_cv_noisy = model.cross_val_score(5)

    print(f'Clean Data Test Accuracy: {acc_clean:.4f}')
    print(f'Clean Data {k}-fold Cross Validation Accuracy: {acc_cv_clean:.4f}')
    print(f'Noisy Data Test Accuracy: {acc_noisy:.4f}')
    print(f'Noisy Data {k}-fold Cross Validation Accuracy: {acc_cv_noisy:.4f}')


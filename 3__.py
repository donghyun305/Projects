from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from random import seed, random

class Generate_Data(object):
    def __init__(self, data_size):
        self.data_size = data_size

    def original_data(self):
        seed(1)
        x = [2 * random() - 1 for _ in range(self.data_size)]
        y = [2 * random() - 1 for _ in range(self.data_size)]
        class_label = [1 if 0.5 * i + j > 0 else -1 for i, j in zip(x, y)]
        return list(zip(class_label, x))

class NoisySVM:
    def __init__(self, X, y, rho_p, rho_n):
        self.X = X
        self.y = y
        self.rho_p = rho_p
        self.rho_n = rho_n

    def add_noise(self, X, y):
        n_samples = len(y)
        n_noisy_p = int(n_samples * self.rho_p)
        n_noisy_n = int(n_samples * self.rho_n)

        noisy_indices_p = np.random.choice(np.where(y == 1)[0], n_noisy_p, replace=False)
        noisy_indices_n = np.random.choice(np.where(y == -1)[0], n_noisy_n, replace=False)

        y_noisy = y.copy()
        y_noisy[noisy_indices_p] = -1
        y_noisy[noisy_indices_n] = 1

        return X, y_noisy

    def train_clean_model(self):
        model = SVC()
        self.X = np.reshape(self.X, (-1, 1))
        self.y = np.ravel(self.y)  # Use np.ravel() here

        model.fit(self.X, self.y)
        return model

    def train_noisy_model(self, X_noisy, y_noisy):
        weights = self.calculate_weights(X_noisy, y_noisy)
        model = SVC()
        X_noisy = X_noisy.reshape(-1, 1)
        y_noisy = np.ravel(y_noisy)  # Use np.ravel() here

        model.fit(X_noisy, y_noisy, sample_weight=weights)
        return model

    def calculate_weights(self, X_noisy, y_noisy):
        clean_model = self.train_clean_model()
        y_pred = clean_model.predict(X_noisy.reshape(-1, 1))

        weights = np.zeros(y_noisy.shape[0])
        for i in range(y_noisy.shape[0]):
            if y_pred[i] == y_noisy[i]:
                weights[i] = self.rho_p / (self.rho_p + self.rho_n)
            else:
                weights[i] = self.rho_n / (self.rho_p + self.rho_n)

        return weights

gen_data = Generate_Data(data_size=500)
data = gen_data.original_data()

data = np.array(data)

X = data[:, 1]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

rho_p = 0.3
rho_n = 0.3

noisy_svm = NoisySVM(X_train, y_train, rho_p, rho_n)
clean_model = noisy_svm.train_clean_model()
X_noisy_train, y_noisy_train = noisy_svm.add_noise(X_train, y_train)
X_noisy_train = np.reshape(X_noisy_train, (-1, 1))

noisy_model = noisy_svm.train_noisy_model(X_noisy_train, y_noisy_train)

# Calculate accuracy and cross-validation scores
clean_accuracy = clean_model.score(X_test, y_test)
noisy_accuracy = noisy_model.score(X_test, y_test)

clean_cv_score = np.mean(cross_val_score(clean_model, X, y, cv=5))
noisy_cv_score = np.mean(cross_val_score(noisy_model, X, y, cv=5))

print(f"Clean model accuracy: {clean_accuracy:.3f}")
print(f"Clean model cross-validation score: {clean_cv_score:.3f}")
print(f"Noisy model accuracy: {noisy_accuracy:.3f}")
print(f"Noisy model cross-validation score: {noisy_cv_score:.3f}")









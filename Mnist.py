import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from random import seed, random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def add_noise(X, y, rho_p, rho_n):
    n_samples = len(y)
    n_noisy_p = int(n_samples * rho_p)
    n_noisy_n = int(n_samples * rho_n)

    noisy_indices_p = np.random.choice(np.where(y == 1)[0], n_noisy_p, replace=False)
    noisy_indices_n = np.random.choice(np.where(y == 0)[0], n_noisy_n, replace=False)

    y_noisy = y.copy()
    y_noisy[noisy_indices_p] = 0
    y_noisy[noisy_indices_n] = 1

    return X, y_noisy

def train_clean_model(X, y):
    model = SVC()
    model.fit(X, y)
    return model

def train_noisy_model(X, y, X_noisy, y_noisy):
    clean_model = train_clean_model(X, y)
    y_pred = clean_model.predict(X_noisy)
    weights = np.zeros(y_noisy.shape[0])
    for i in range(y_noisy.shape[0]):
        if y_pred[i] == y_noisy[i]:
            weights[i] = rho_p / (rho_p + rho_n)
        else:
            weights[i] = rho_n / (rho_p + rho_n)

    model = SVC()
    model.fit(X_noisy, y_noisy, sample_weight=weights)
    return model

clean_svm_model = train_clean_model(X_train, y_train)
clean_svm_accuracy = clean_svm_model.score(X_test, y_test)
clean_svm_cv_scores = cross_val_score(clean_svm_model, X, y, cv=5)
print("Clean SVM model accuracy:", clean_svm_accuracy)
print("Clean SVM model cross validation scores:", clean_svm_cv_scores)

rho_p = 0.2
rho_n = 0.2
X_noisy_train, y_noisy_train = add_noise(X_train, y_train, rho_p, rho_n)

noisy_svm_model = train_noisy_model(X_train, y_train, X_noisy_train, y_noisy_train)

noisy_svm_accuracy = noisy_svm_model.score(X_test, y_test)
noisy_svm_cv_scores = cross_val_score(noisy_svm_model, X, y, cv=5)
print("Noisy SVM model accuracy:", noisy_svm_accuracy)
print("Noisy SVM model cross validation scores:", noisy_svm_cv_scores)




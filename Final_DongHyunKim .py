import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from random import seed, random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# 1. Generating linearly separable data
def generate_data(data_size):
    seed(1)
    x = [2 * random() - 1 for _ in range(data_size)]
    y = [2 * random() - 1 for _ in range(data_size)]
    class_label = [1 if 0.5 * i + j > 0 else -1 for i, j in zip(x, y)]
    return np.array(list(zip(x, y))), np.array(class_label)

# 2. Adding Noise to the data
def add_noise(X, y, rho_p, rho_n):
    n_samples = len(y)
    n_noisy_p = int(n_samples * rho_p)
    n_noisy_n = int(n_samples * rho_n)

    noisy_indices_p = np.random.choice(np.where(y == 1)[0], n_noisy_p, replace=False)
    noisy_indices_n = np.random.choice(np.where(y == -1)[0], n_noisy_n, replace=False)

    y_noisy = y.copy()
    y_noisy[noisy_indices_p] = -1
    y_noisy[noisy_indices_n] = 1

    return X, y_noisy

# 3. Models for Clean data
def train_clean_model(X, y):
    model = SVC()
    model.fit(X, y)
    return model

def train_clean_lr_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def train_clean_rf_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# 4. Models for noisy labeled data
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

def train_noisy_lr_model(X, y, X_noisy, y_noisy):
    clean_model = train_clean_lr_model(X, y)
    y_pred = clean_model.predict(X_noisy)

    weights = np.zeros(y_noisy.shape[0])
    for i in range(y_noisy.shape[0]):
        if y_pred[i] == y_noisy[i]:
            weights[i] = rho_p / (rho_p + rho_n)
        else:
            weights[i] = rho_n / (rho_p + rho_n)

    model = LogisticRegression()
    model.fit(X_noisy, y_noisy, sample_weight=weights)
    return model

def train_noisy_rf_model(X, y, X_noisy, y_noisy):
    clean_model = train_clean_rf_model(X, y)
    y_pred = clean_model.predict(X_noisy)

    weights = np.zeros(y_noisy.shape[0])
    for i in range(y_noisy.shape[0]):
        if y_pred[i] == y_noisy[i]:
            weights[i] = rho_p / (rho_p + rho_n)
        else:
            weights[i] = rho_n / (rho_p + rho_n)

    model = RandomForestClassifier()
    model.fit(X_noisy, y_noisy, sample_weight=weights)
    return model



data_size = 500
X, y = generate_data(data_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rho_p = 0.2
rho_n = 0.2
X_noisy_train, y_noisy_train = add_noise(X_train, y_train, rho_p, rho_n)

clean_svm_model = train_clean_model(X_train, y_train)
clean_lr_model = train_clean_lr_model(X_train, y_train)
clean_rf_model = train_clean_rf_model(X_train, y_train)

noisy_svm_model = train_noisy_model(X_train, y_train, X_noisy_train, y_noisy_train)
noisy_lr_model = train_noisy_lr_model(X_train, y_train, X_noisy_train, y_noisy_train)
noisy_rf_model = train_noisy_rf_model(X_train, y_train, X_noisy_train, y_noisy_train)

clean_svm_accuracy = clean_svm_model.score(X_test, y_test)
clean_svm_cv_scores = cross_val_score(clean_svm_model, X, y, cv=5)
print("Clean SVM model accuracy:", clean_svm_accuracy)
print("Clean SVM model cross validation scores:", clean_svm_cv_scores)

clean_lr_accuracy = clean_lr_model.score(X_test, y_test)
clean_lr_cv_scores = cross_val_score(clean_lr_model, X, y, cv=5)
print("Clean LR model accuracy:", clean_lr_accuracy)
print("Clean LR model cross validation scores:", clean_lr_cv_scores)

clean_rf_accuracy = clean_rf_model.score(X_test, y_test)
clean_rf_cv_scores = cross_val_score(clean_rf_model, X, y, cv=5)
print("Clean RF model accuracy:", clean_rf_accuracy)
print("Clean RF model cross validation scores:", clean_rf_cv_scores)


noisy_svm_accuracy = noisy_svm_model.score(X_test, y_test)
noisy_svm_cv_scores = cross_val_score(noisy_svm_model, X, y, cv=5)
print("Noisy SVM model accuracy:", noisy_svm_accuracy)
print("Noisy SVM model cross validation scores:", noisy_svm_cv_scores)

noisy_lr_accuracy = noisy_lr_model.score(X_test, y_test)
noisy_lr_cv_scores = cross_val_score(noisy_lr_model, X, y, cv=5)
print("Noisy LR model accuracy:", noisy_lr_accuracy)
print("Noisy LR model cross validation scores:", noisy_lr_cv_scores)


noisy_rf_accuracy = noisy_rf_model.score(X_test, y_test)
noisy_rf_cv_scores = cross_val_score(noisy_rf_model, X, y, cv=5)
print("Noisy RF model accuracy:", noisy_rf_accuracy)
print("Noisy RF model cross validation scores:", noisy_rf_cv_scores)

def plot_decision_boundary(X, y, model, ax, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=70)
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    return scatter

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
scatter1 = plot_decision_boundary(X, y, clean_svm_model, ax[0], "Decision Boundary (Clean Model)")
scatter2 = plot_decision_boundary(X, y, noisy_svm_model, ax[1], "Decision Boundary (Noisy Model)")
plt.legend(*scatter1.legend_elements(), loc="lower right", title="Classes")
plt.show()
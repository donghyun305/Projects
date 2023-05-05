import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from functools import partial
from random import random, seed
from final_main import Generate_Data

class Training_noisy_model(Generate_Data.random_data(object)):
    def __init__(self, data_size, is_random, rho_p, rho_n, dim=2, ws):
        self.data = Generate_Data(data_size)
        self.true_data_map = {}
        if is_random:
            noise_free_data, self.n1, self.n2




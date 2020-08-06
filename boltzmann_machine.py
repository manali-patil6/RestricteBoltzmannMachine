import numpy as np
import pandas as pd
import torch
import torch.utils.data

from .helpers import convert, trainer, tester
from .rbm_structure import RBM

# Importing Data
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Data Manipulation

training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting number of users and movies

nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

# Restructuring data
training_set = convert(training_set, nb_users, nb_movies)
test_set = convert(test_set, nb_users, nb_movies)

# Initializing Torch tensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting to binary ranking

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the RBM from previously made structure

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM

nb_epoch = 10
trainer(nb_epoch, nb_users, batch_size, training_set, rbm)

# Testing

tester(nb_users, test_set, training_set, rbm)

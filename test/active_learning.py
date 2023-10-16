import sys
sys.path.append('../src')

import numpy as np
import torch
import time
import sys
import resource
import copy
from torch.utils.data import DataLoader

from data_classes import *
from read_input import *
from read_trainset import *
from network import *
from prepare_batches import *
from traininit import *
from data_set import *
from data_loader import *
from optimization_step import *
from output_nn import *
from py_aeio import *
from bnn import BayesianNeuralNetwork
from bnn import get_batch

device = "cpu"
tin_file = "train.in"
tin = read_train_in(tin_file)
torch.manual_seed(3)
np.random.seed(tin.numpy_seed)
tin.train_forces = False

tin.train_file = 'Cu.active_learning'
list_structures_energy, _, list_removed, max_nnb, tin = read_list_structures(tin)

net = NetAtom(tin.networks_param["input_size"], tin.networks_param["hidden_size"],
			    tin.sys_species, tin.networks_param["activations"], tin.alpha, device)

bnn = BayesianNeuralNetwork(net)

np.random.seed(42)
dataset_size = len(list_structures_energy)
indices = list(range(dataset_size))
np.random.shuffle(indices)

training_indices = indices[:5000]
test_indices = indices[5000:6000]
valid_indices = indices[6000:]

training_structures_energy = [list_structures_energy[x] for x in training_indices]
test_structures_energy     = [list_structures_energy[x] for x in test_indices]
valid_structure_energy     = [list_structures_energy[x] for x in valid_indices]

training_batch = get_batch(tin, training_structures_energy, max_nnb)
test_batch     = get_batch(tin, test_structures_energy, max_nnb)
valid_batch    = get_batch(tin, valid_structure_energy, max_nnb)

EPOCHS = 1
NUM_SAMPLES = 10000
LR = 0.01

bnn.train(training_batch, EPOCHS, initial_lr=LR, verbose=True)

valid_pred = bnn.predict(valid_batch,num_samples=NUM_SAMPLES)
std_valid_batch = torch.std(valid_pred['obs'],0)

test_pred = bnn.predict(test_batch,num_samples=NUM_SAMPLES)
std_test_batch = torch.std(test_pred['obs'],0)
idx_test_sorted = np.argsort(std_test_batch)

l2 = bnn.get_loss_RMSE(valid_batch, num_samples=NUM_SAMPLES)
print('RMSD valid set pre train {}'.format(l2))

with open('std_test_multi.txt', 'w') as out:
    for i in range(0, 10): 
    
        l = list(range(i*100,i*100+10))

        indices = [test_indices[x] for x in idx_test_sorted[l]]
        std_new_points = [std_test_batch[x] for x in idx_test_sorted[l]]
        new_structure_energy = [list_structures_energy[x] for x in indices]
        new_training_structures_energy = training_structures_energy + new_structure_energy
        new_training_batch = get_batch(tin, new_training_structures_energy, max_nnb)

        bnn1 = copy.deepcopy(bnn)
        bnn1.train(new_training_batch, EPOCHS, initial_lr=LR, verbose=False)

        valid_pred = bnn1.predict(valid_batch,num_samples=NUM_SAMPLES)
        std_valid_batch = torch.mean(torch.std(valid_pred['obs'],0))
        l2 = bnn1.get_loss_RMSE(valid_batch, num_samples=NUM_SAMPLES)
        out.write('{} {} {}\n'.format(std_new_points, std_valid_batch, l2))
        print(l2, std_new_points, std_valid_batch)

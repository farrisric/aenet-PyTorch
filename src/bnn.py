from pyro.nn.module import to_pyro_module_
from pyro.nn import PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer import Predictive
import pyro
import torch
import numpy as np
import torch
import time
import sys
import resource
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

class BayesianNeuralNetwork():
    def __init__(self, net):
        self.net = net
        self.species = net.species
        self.device = net.device
        self.guide = None
        self.epochs = 0
        to_pyro_module_(self.net)
        self.initialize()

    def initialize(self):
        for m in self.net.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                if name == 'weight':
                    setattr(m, name, PyroSample(prior=dist.Normal(0, 1)
                                                    .expand(value.shape)
                                                    .to_event(value.dim())))
                if name == 'bias':
                    setattr(m, name, PyroSample(prior=dist.Normal(0, 10)
                                                    .expand(value.shape)
                                                    .to_event(value.dim())))
        self.guide = AutoDiagonalNormal(self.model)


    def model(self, grp_descrp, logic_reduce, grp_energy=None):    
        partial_E_ann = [0 for i in range(len(self.species))]
        
        for iesp in range(len(self.species)):
            partial_E_ann[iesp] = self.net.functions[iesp](grp_descrp[iesp])
            
        sigma = pyro.sample('noise', dist.Uniform(0,10))
        with pyro.plate('data', len(logic_reduce[0])):

            list_E_ann = torch.zeros((len(logic_reduce[0])), device=self.device ).double()
            for iesp in range(len(self.species)):
                list_E_ann = list_E_ann + torch.einsum( "ij,ki->k", partial_E_ann[iesp], logic_reduce[iesp] )
            pyro.sample('obs', dist.Normal(list_E_ann, sigma), obs=grp_energy)
        return list_E_ann

    def train(self, grouped_train_loader ,epochs, initial_lr = 0.001, verbose=False):
        pyro.clear_param_store()
        gamma = 0.1  # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / epochs)
        optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
        svi = SVI(self.model, self.guide, optim, loss=TraceMeanField_ELBO())
        losses = []
        for j in range(epochs):
            self.epochs += 1
            loss = 0
            for data_batch in grouped_train_loader:
                grp_descrp, grp_energy, logic_reduce = data_batch[0][10], data_batch[0][11], data_batch[0][12]
                
                for i in range(len(grp_descrp)):
                    grp_descrp[i] = grp_descrp[i].float()

                for i in range(len(logic_reduce)):
                    logic_reduce[i] = logic_reduce[i].float()

                loss += svi.step(grp_descrp, logic_reduce, grp_energy) 
            if verbose:
                if j % 100 == 0:
                    print("[EPOCH LOSS %04d] loss: %.4f" % (j + 1, loss / len(logic_reduce[0])))

                if j % 1000 == 0:
                    l2, _ = self.get_loss_RMSE(grouped_train_loader, num_samples=800)
                    print("[EPOCH RMSD %04d] loss: %.4f" % (j + 1, l2))
        return loss / len(logic_reduce[0])
    
    def predict(self, grouped_loader, num_samples=8000):
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples,
                        return_sites=("obs", "_RETURN"))
        grp_descrp, grp_energy, logic_reduce, grp_N_atoms = get_train_test(grouped_loader)
        
        return predictive(grp_descrp, logic_reduce)
    
    def get_loss_RMSE(self, grouped_loader, num_samples=800):
        """
		[Energy training] Compute root mean squared error of energies in the batch
		"""
        grp_descrp, grp_energy, logic_reduce, grp_N_atoms = get_train_test(grouped_loader)
        list_E_ann = self.predict(grouped_loader, num_samples=num_samples)['obs']
        differences = (list_E_ann - grp_energy)

        l2s = torch.sum( differences**2/grp_N_atoms**2, dim=1)
        l2 = torch.mean(l2s)
        std_l2 = torch.std(l2s)
        
        return l2, std_l2

def get_batch(tin, list_structures_energy, max_nnb):
    dataset_energy = StructureDataset(list_structures_energy, tin.sys_species, tin.networks_param["input_size"], max_nnb)
        
    dataset_energy_size = len(dataset_energy)

    # Normalize
    # E_scaling, E_shift = tin.trainset_params.E_scaling, tin.trainset_params.E_shift
    # sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
    # dataset_energy.normalize_E(tin.trainset_params.E_scaling,tin.trainset_params.E_shift)
    # stp_shift, stp_scale = dataset_energy.normalize_stp(sfval_avg, sfval_cov)

    energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, sampler=range(len(dataset_energy)), N_batch=1)   

    grouped_data = GroupedDataset(energy_data, None,)
    
    grouped_loader = DataLoader(grouped_data, batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, num_workers=0) 
    return grouped_loader

def select_batches(tin, trainset_params, device, list_structures_energy, list_structures_forces,
				   max_nnb, N_batch_train, N_batch_test, N_batch_valid, train_sampler_E, test_sampler_E, valid_sampler_E):
        """
        Select which structures belong to each batch for training.
        Returns: four objects of the class data_set_loader.PrepDataloader(), for train/test and energy/forces
        """
        dataset_energy = StructureDataset(list_structures_energy, tin.sys_species, tin.networks_param["input_size"], max_nnb)
        
        dataset_energy_size = len(dataset_energy)

        # Normalize
        E_scaling, E_shift = tin.trainset_params.E_scaling, tin.trainset_params.E_shift
        sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
        dataset_energy.normalize_E(trainset_params.E_scaling, trainset_params.E_shift)
        stp_shift, stp_scale = dataset_energy.normalize_stp(sfval_avg, sfval_cov)

        # Split in train/test
        #train_sampler_E, valid_sampler_E = split_database(dataset_energy_size, tin.test_split)

        train_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_train,
                                        sampler=train_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="train_energy")
        test_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_train,
                                        sampler=test_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="test_energy")

        if valid_sampler_E:
            valid_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_valid,
                                        sampler=valid_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="valid_energy")
            return train_energy_data, test_energy_data, valid_energy_data
        
        return train_energy_data, test_energy_data

def get_sets_from_indices(tin, max_nnb, training_indices, test_indices, valid_indices, list_structures_energy):
    N_batch_train, N_batch_test, N_batch_valid = 1, 1, 1 #select_batch_size(tin, list_structures_energy, list_structures_forces)
    device = 'cpu'
    # Join datasets with forces and only energies in a single torch dataset AND prepare batches
    if valid_indices:
        train_energy_data, test_energy_data, valid_energy_data = select_batches(tin, tin.trainset_params, device, list_structures_energy, None,
                    max_nnb, N_batch_train,N_batch_test, N_batch_valid, training_indices, test_indices, valid_indices)
    else:
        train_energy_data, test_energy_data = select_batches(tin, tin.trainset_params, device, list_structures_energy, None,
                    max_nnb, N_batch_train,N_batch_test, N_batch_valid, training_indices, test_indices, valid_indices)
        
    grouped_train_data = GroupedDataset(train_energy_data, None,
									 memory_mode=tin.memory_mode, device=device, dataname="train")
    grouped_test_data = GroupedDataset(test_energy_data, None,
									 memory_mode=tin.memory_mode, device=device, dataname="test")
    
    grouped_train_loader = DataLoader(grouped_train_data, batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, num_workers=0)
    grouped_test_loader = DataLoader(grouped_test_data, batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, num_workers=0)

    if valid_indices:
        grouped_valid_data = GroupedDataset(valid_energy_data, None,
									    memory_mode=tin.memory_mode, device=device, dataname="valid")
        grouped_valid_loader = DataLoader(grouped_valid_data, batch_size=1, shuffle=False,
                                    collate_fn=custom_collate, num_workers=0)
        
        return grouped_train_loader, grouped_test_loader, grouped_valid_loader
    
    return grouped_train_loader, grouped_test_loader

def get_train_test(grouped_loader):
    for data_batch in grouped_loader:
        grp_descrp, grp_energy, logic_reduce, grp_N_atoms = data_batch[0][10], data_batch[0][11], data_batch[0][12],data_batch[0][14]

    for i in range(len(grp_descrp)):
        grp_descrp[i] = grp_descrp[i].float()

    for i in range(len(logic_reduce)):
        logic_reduce[i] = logic_reduce[i].float()

    return grp_descrp, grp_energy, logic_reduce, grp_N_atoms
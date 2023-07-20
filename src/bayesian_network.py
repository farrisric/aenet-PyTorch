import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from network import NetAtom

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroSample
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from pyro.nn.module import to_pyro_module_
from pyro.distributions import constraints		
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO

class BayesianNetAtoms(PyroModule):
	def __init__(self, net):
		super().__init__()
		self.species = net.species
		self.device = net.device
		self.model = net
		to_pyro_module_(self.model)
		for m, iets in zip(self.model.modules(), self.species):
			for i, name, value in enumerate(list(m.named_parameters(recurse=False))):
				if name == 'weight':
					setattr(m, f'{iets}_h{i}_{name}', PyroSample(prior=dist.Normal(0, 1)
													.expand(value.shape)
													.to_event(value.dim())))
				if name == 'bias':
					setattr(m, f'{iets}_h{i}_{name}', PyroSample(prior=dist.Normal(0, 1)
													.expand(value.shape)
													.to_event(value.dim())))
	# def model(self, grp_descrp, logic_reduce, y):
	# 	pass

	def initialize(self):
		self.optim = pyro.optim.Adam({"lr": 0.05})
		self.loss = TraceGraph_ELBO()
		self.svi = SVI(self.model, self.guide, self.optim, loss=self.loss)

	def step(self, grp_descrp, logic_reduce, grp_energy):
		return self.svi.step(grp_descrp, logic_reduce, grp_energy)
	
	def guide(self, grp_descrp, logic_reduce, grp_energy=None):
		for m, iets in zip(self.model.modules(), self.species):
			for i, name, value in enumerate(list(m.named_parameters(recurse=False))):
				if name == 'weight':
					setattr(m, f'{iets}_h{i}_{name}', PyroSample(prior=dist.Normal(0, 1)
													.expand(value.shape)
													.to_event(value.dim())))
				if name == 'bias':
					setattr(m, f'{iets}_h{i}_{name}', PyroSample(prior=dist.Normal(0, 1)
													.expand(value.shape)
													.to_event(value.dim())))

		partial_E_ann = [0 for i in range(len(self.species))]
		sigmas = {iets: pyro.sample(f'sigma_{iets}', dist.Uniform(0., 10.)) for iets in self.species}

		# Local Energies calculation per species
		for iesp in range(len(self.species)):
			partial_E_ann[iesp] = self.model[iesp](grp_descrp[iesp])

		new_partial_E_ann = [0 for i in range(len(self.species))]
		for i, iets in enumerate(self.species):
			with pyro.plate(f'data_{iets}', len(grp_descrp[iesp])):
				new_partial_E_ann[i] = pyro.sample(f'local_{iets}', dist.Normal(partial_E_ann[iesp], sigmas[iets]))
		
		list_E_ann = torch.zeros((len(logic_reduce[0])), device=self.device)
		for iesp in range(len(self.species)):
			list_E_ann = list_E_ann + torch.einsum( "ij,ki->k", new_partial_E_ann[iesp], logic_reduce[iesp])

		sigma_tot = pyro.sample('sigma_tot', dist.Uniform(0., 10.))
		

	def forwardino(self, grp_descrp, logic_reduce, grp_energy=None):
		partial_E_ann = [0 for i in range(len(self.species))]
		sigmas = {f'sigma_{iets}': pyro.sample(iets, dist.Uniform(0., 10.)) for iets in self.species}

		# Local Energies calculation per species
		for iesp in range(len(self.species)):
			partial_E_ann[iesp] = self.model[iesp](grp_descrp[iesp])

		new_partial_E_ann = [0 for i in range(len(self.species))]
		for i, iets in enumerate(self.species):
			with pyro.plate(f'data_{iets}', len(grp_descrp[iesp])):
				new_partial_E_ann[i] = pyro.sample(f'local_{iets}', dist.Normal(partial_E_ann[iesp], sigmas[f'sigma_{iets}']))
		
		list_E_ann = torch.zeros((len(logic_reduce[0])), device=self.device)
		for iesp in range(len(self.species)):
			list_E_ann = list_E_ann + torch.einsum( "ij,ki->k", new_partial_E_ann[iesp], logic_reduce[iesp])

		sigma_tot = pyro.sample('sigma_tot', dist.Uniform(0., 10.))
		with pyro.plate('total_energy', len(logic_reduce[0])):
			pyro.sample('obs', dist.Normal(list_E_ann, sigma_tot), obs=grp_energy)				
		return list_E_ann

		# with pyro.plate('total_energy', len(grp_energy)):
		# 	total_energy = pyro.sample('obs', dist.Normal(list_E_ann, sigma), obs=grp_energy)
		# # sigmas = {pyro.sample(f'sigma_{iets}', dist.Uniform(0., 10.)) for iets in self.species}
		# # local_Es = {iets : None for iets in self.species}
		# # for i, iets in enumerate(self.species):
		# # 	with pyro.plate(f'data_{iets}', len(grp_descrp[i].shape[0])):
		# # 		local_Es[iets] = pyro.sample(f'local_{iets}', dist.Normal(partial_E_ann[iesp], sigmas[f'sigma_{iets}']))

		# list_E_ann = torch.zeros((len(logic_reduce[0])), device=self.device)
		# for iesp in range(len(self.species)):
		# 	list_E_ann = list_E_ann + torch.einsum( "ij,ki->k", partial_E_ann[iesp], logic_reduce[iesp] )

		# sigma = pyro.sample('sigma', dist.Uniform(0., 10.))
		# with pyro.plate('data', len(grp_energy)):
		# 	obs = pyro.sample('obs', dist.Normal(list_E_ann, sigma), obs=grp_energy)
		# return list_E_ann

if __name__ == '__main__':
	model = NetAtom(input_size=[20,20], hidden_size=[[5,5],[15,15]], species=['Pd', 'O'], activations=[['tanh','tanh'],['tanh','tanh']], alpha=0.1, device='cpu')

	bayesian_model = BayesianNetAtoms(model)
	print(bayesian_model)
	
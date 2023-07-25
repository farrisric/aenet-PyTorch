from pyro.nn.module import to_pyro_module_
from pyro.nn import PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
import pyro

from network import NetAtom

net_atom = NetAtom()

to_pyro_module_(net_atom)

def model_label(net_atom):
    weight_labels = []
    bias_labels = []
    for specie, net in zip(net_atom.species, net_atom.hidden_size):
        for h_i in range(len(net)):
            weight_labels.append(f'{specie}_h{h_i}_weight')
            bias_labels.append(f'{specie}_h{h_i}_bias')
            
        weight_labels.append(f'{specie}_h{h_i+1}_weight')
        bias_labels.append(f'{specie}_h{h_i+1}_bias')
        
    return weight_labels, bias_labels

weight_labels, bias_labels = model_label(net_atom)

i = 0
for m in net_atom.modules():
    for name, value in list(m.named_parameters(recurse=False)):
        if name == 'weight':
            setattr(m, weight_labels[i], PyroSample(prior=dist.Normal(0, 1)
                                              .expand(value.shape)
                                              .to_event(value.dim())))
        if name == 'bias':
            setattr(m, bias_labels[i], PyroSample(prior=dist.Normal(0, 10)
                                              .expand(value.shape)
                                              .to_event(value.dim())))
            print(weight_labels[i])
            i += 1
                                              
def model(grp_descrp, logic_reduce, grp_energy=None):
    mean = net_atom.forward(grp_descrp, logic_reduce)
    sigma = pyro.sample('noise', dist.Uniform(0,10))
    with pyro.plate('data', len(logic_reduce[0])):
        pyro.sample('obs', dist.Normal(mean, sigma), obs=grp_energy)
    return mean

def guide(model):
    return AutoDiagonalNormal(model)

def train(grp_descrp, logic_reduce, grp_energy=None):
	pyro.clear_param_store()
	adam = pyro.optim.Adam({"lr": 0.05})
	svi = SVI(model, guide, adam, loss=Trace_ELBO())
	for x in range(5000):
		if x % 1000:
			print(svi.step(grp_descrp, logic_reduce, grp_energy=None))
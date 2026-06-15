from ddh.losses import total_loss
from ddh.dsm_compat import get_optimizer, _reshape_tensor_with_nans

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def train_ddh(model,
			  x_train, t_train, e_train,
			  x_valid, t_valid, e_valid,
			  alpha, beta, sigma,
			  n_iter = 10000, lr = 1e-3,
			  bs = 100, cuda = False,
			  cause_weights = None,
			  patience_limit = 10,
			  grad_clip_norm = 1.0):

	optimizer = get_optimizer(model, lr)
	# Halve LR when val loss plateaus; fires at half the early-stop patience so
	# the model gets a second chance at the lower rate before training ends.
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.5,
		patience=max(1, patience_limit // 2), min_lr=1e-6,
	)

	patience, old_loss = 0, np.inf
	best_param = deepcopy(model.state_dict())
	n_train = x_train.shape[0]

	for i in tqdm(range(n_iter)):
		# Re-shuffle every epoch so batch composition varies across epochs.
		perm = torch.randperm(n_train)
		x_ep = x_train[perm]
		t_ep = t_train[perm]
		e_ep = e_train[perm]

		model.train()
		for j in range(0, n_train, bs):
			xb = x_ep[j : j + bs]
			tb = t_ep[j : j + bs]
			eb = e_ep[j : j + bs]

			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

			optimizer.zero_grad()
			loss = total_loss(model, xb, tb, eb, alpha, beta, sigma, cause_weights=cause_weights)
			loss.backward()
			if grad_clip_norm > 0:
				nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
			optimizer.step()

		# Validation — skip gradient tracking to save memory and compute.
		model.eval()
		n_val = x_valid.shape[0]
		valid_loss = 0.0
		n_val_batches = 0
		with torch.no_grad():
			for j in range(0, n_val, bs):
				xb = x_valid[j : j + bs]
				tb = t_valid[j : j + bs]
				eb = e_valid[j : j + bs]

				if xb.shape[0] == 0:
					continue

				if cuda:
					xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

				valid_loss += total_loss(model, xb, tb, eb, alpha, beta, sigma, cause_weights=cause_weights).item()
				n_val_batches += 1

		# Normalize by batch count so comparisons are consistent across epochs.
		valid_loss /= max(n_val_batches, 1)
		scheduler.step(valid_loss)

		if valid_loss < old_loss:
			patience = 0
			old_loss = valid_loss
			best_param = deepcopy(model.state_dict())
		else:
			patience += 1
			if patience > patience_limit:
				break

	model.load_state_dict(best_param)
	return model

def create_nn(inputdim, outputdim, dropout = 0.6, layers = [100, 100], activation = 'ReLU', no_activation_last = False):
	modules = []
	if dropout > 0:
		modules.append(nn.Dropout(p = dropout))

	if activation == 'ReLU6':
		act = nn.ReLU6()
	elif activation == 'ReLU':
		act = nn.ReLU()
	elif activation == 'SeLU':
		act = nn.SELU()
	elif activation == 'Tanh':
		act = nn.Tanh()

	prevdim = inputdim
	for hidden in layers + [outputdim]:
		modules.append(nn.Linear(prevdim, hidden, bias = True))
		modules.append(act)
		prevdim = hidden

	if no_activation_last:
		modules = modules[:-1]

	return nn.Sequential(*modules)

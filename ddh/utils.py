from ddh.losses import total_loss
from ddh.dsm_compat import get_optimizer, _reshape_tensor_with_nans

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def _augment_landmark_truncation(x_ep, s_ep, visit_days_ep, split_time, prob):
	"""
		Simulate landmark evaluation during training: for a random subset of
		patients with >=2 observed visits, truncate their sequence to a random
		earlier prefix and re-bin their landmark index `s` to match, instead of
		always training on each patient's full observed history. Mutates and
		returns x_ep/s_ep in place; patients not selected are left untouched.
	"""
	if prob <= 0 or visit_days_ep is None or split_time is None:
		return x_ep, s_ep

	n_obs = (~torch.isnan(x_ep[:, :, 0])).sum(dim = 1)
	eligible = (n_obs >= 2).nonzero(as_tuple = True)[0]
	if eligible.numel() == 0:
		return x_ep, s_ep

	max_bin = split_time.numel() - 2
	selected = eligible[torch.rand(eligible.numel(), device = eligible.device) < prob]
	for i in selected.tolist():
		n = int(n_obs[i].item())
		k = torch.randint(1, n, (1,)).item() # keep k of the n observed visits
		x_ep[i, k:, :] = float('nan')

		day_k = visit_days_ep[i, k - 1]
		if torch.isnan(day_k):
			continue
		bin_idx = int(torch.bucketize(day_k, split_time, right = True).item()) - 1
		s_ep[i] = max(0, min(bin_idx, max_bin))

	return x_ep, s_ep

def train_ddh(model,
			  x_train, t_train, e_train,
			  x_valid, t_valid, e_valid,
			  alpha, beta, sigma,
			  n_iter = 10000, lr = 1e-3,
			  bs = 100, cuda = False,
			  s_train = None, s_valid = None,
			  cause_weights = None,
			  patience_limit = 10,
			  grad_clip_norm = 1.0,
			  visit_days_train = None,
			  split_time = None,
			  landmark_augment_prob = 0.0):

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
	augment = landmark_augment_prob > 0 and visit_days_train is not None and s_train is not None and split_time is not None
	split_time_tensor = torch.as_tensor(split_time).double() if augment else None

	for i in tqdm(range(n_iter)):
		# Re-shuffle every epoch so batch composition varies across epochs.
		perm = torch.randperm(n_train)
		x_ep = x_train[perm]
		t_ep = t_train[perm]
		e_ep = e_train[perm]
		s_ep = s_train[perm] if s_train is not None else None

		if augment:
			visit_days_ep = visit_days_train[perm]
			x_ep, s_ep = _augment_landmark_truncation(x_ep, s_ep, visit_days_ep, split_time_tensor, landmark_augment_prob)

		model.train()
		for j in range(0, n_train, bs):
			xb = x_ep[j : j + bs]
			tb = t_ep[j : j + bs]
			eb = e_ep[j : j + bs]
			sb = s_ep[j : j + bs] if s_ep is not None else None

			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
				if sb is not None:
					sb = sb.cuda()

			optimizer.zero_grad()
			loss = total_loss(model, xb, tb, eb, alpha, beta, sigma, s=sb, cause_weights=cause_weights)
			loss.backward()
			if grad_clip_norm > 0:
				nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
			optimizer.step()

		# Validation — skip gradient tracking to save memory and compute.
		model.eval()
		with torch.no_grad():
			xb, tb, eb = x_valid, t_valid, e_valid
			sb = s_valid

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
				if sb is not None:
					sb = sb.cuda()

			valid_loss = total_loss(
				model, xb, tb, eb, alpha, beta, sigma,
				s=sb, cause_weights=cause_weights
			).item()
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

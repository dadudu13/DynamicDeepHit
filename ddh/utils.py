from ddh.losses import (total_loss, build_landmark_prefixes, landmark_nll_loss,
                        landmark_nll_from_outcomes, conditional_cif, RankRows,
                        weighted_ranking_loss)
from ddh.dsm_compat import get_optimizer, _reshape_tensor_with_nans

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.checkpoint import checkpoint

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

def _chunked_outcomes(model, x_exp, chunk_rows, recompute):
	"""
		Model outputs for every expanded landmark row, one chunk at a time.

		With recompute=True each chunk's forward is wrapped in a gradient
		checkpoint: its activations are freed immediately and recomputed during
		backward, so peak memory tracks `chunk_rows` even though the returned
		tensors are all live at once.  Checkpointing restores the RNG state
		before recomputing, so dropout masks match the original forward and the
		gradient is exact.
	"""
	n_rows = x_exp.shape[0]
	step = n_rows if not chunk_rows or chunk_rows <= 0 else chunk_rows
	chunks = []
	for start in range(0, n_rows, step):
		xc = x_exp[start : start + step]
		if recompute:
			outcomes = checkpoint(lambda z: tuple(model(z)[1]), xc,
								  use_reentrant = False)
		else:
			outcomes = model(xc)[1]
		chunks.append(outcomes)
	return [torch.cat([c[k] for c in chunks], 0) for k in range(len(chunks[0]))]

def all_landmark_objective(model, xb, tb, eb, sb, s_all_b,
						   alpha, beta, sigma, cause_weights,
						   chunk_rows = 512, backward = False,
						   rank_on_landmarks = False):
	"""
		Total loss with the survival term evaluated at EVERY landmark instead of
		only the final visit:

			L = (1-alpha-beta) * L_long + alpha * L_rank
				+ beta * (1/K) sum_i (1/V_i) sum_v -log P(T_i,K_i | T_i>s_iv, X_i,<=v)

		The longitudinal loss stays on the full-history rows.  So does the ranking
		loss by default, which keeps alpha at the scale it was tuned at and leaves
		the beta term as the only change from the final-visit objective.
		Expanding P patients into sum_i V_i prefix rows multiplies activation
		memory by the mean visit count, so the expanded rows are processed in
		chunks of `chunk_rows`; back-propagating each chunk separately is exact
		because the objective is a plain weighted sum over rows (gradients
		accumulate in .grad until the caller steps).

		rank_on_landmarks=True moves the ranking term onto the expanded rows too,
		weighted by the same 1 / V_i:

			L = (1-alpha-beta) * L_long + alpha * L_rank^landmark
				+ beta * (1/K) sum_i (1/V_i) sum_v -log P(...)

		The ranking term couples every row to every other, so it cannot be summed
		chunk by chunk the way the NLL can.  The expanded rows are still forwarded
		in chunks, but through a gradient checkpoint, so `chunk_rows` stays a pure
		memory knob and the objective is unchanged by it — at the cost of a second
		forward pass during backward.

		Returns the scalar loss value.  With backward=True the caller is
		responsible for optimizer.zero_grad()/step() around the call.
	"""
	main = total_loss(model, xb, tb, eb, alpha, beta, sigma,
					  s = sb, cause_weights = cause_weights, include_nll = False,
					  include_rank = not rank_on_landmarks)
	if backward:
		main.backward()
	total = main.item()

	x_exp, s_exp, patient_idx, weight = build_landmark_prefixes(xb, s_all_b)
	t_exp, e_exp = tb.long()[patient_idx], eb.int()[patient_idx]

	if not rank_on_landmarks:
		n_rows = x_exp.shape[0]
		step = n_rows if not chunk_rows or chunk_rows <= 0 else chunk_rows
		for start in range(0, n_rows, step):
			sl = slice(start, start + step)
			nll = beta * landmark_nll_loss(model, x_exp[sl], t_exp[sl], e_exp[sl],
										   s_exp[sl], weight[sl])
			if backward:
				nll.backward()
			total += nll.item()
		return total

	outcomes = _chunked_outcomes(model, x_exp, chunk_rows,
								 recompute = backward and torch.is_grad_enabled())
	cif = [torch.cumsum(ok, 1) for ok in outcomes]

	rows = RankRows(cif = conditional_cif(cif, s_exp), t = t_exp, s = s_exp,
					e = e_exp, weight = weight, patient = patient_idx)
	survival = (
		beta * landmark_nll_from_outcomes(outcomes, cif, t_exp, e_exp, s_exp, weight)
		+ alpha * weighted_ranking_loss(rows, rows, sigma, cause_weights)
	)
	if backward:
		survival.backward()

	return total + survival.item()

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
			  landmark_augment_prob = 0.0,
			  s_all_train = None,
			  s_all_valid = None,
			  all_landmark_nll = False,
			  landmark_chunk_rows = 512,
			  all_landmark_rank = False):

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

	# All-landmark training needs the landmark bin of every timestep, not just of
	# each patient's last visit, and is incompatible with random truncation
	# augmentation (which samples one landmark per epoch — a stochastic subset of
	# what this objective covers deterministically).
	if all_landmark_nll:
		if s_all_train is None:
			raise ValueError("all_landmark_nll=True requires s_all_train")
		if augment:
			raise ValueError("all_landmark_nll=True is incompatible with landmark_augment_prob > 0")
	# The ranking extension reuses the same (patient, visit) expansion, so it only
	# makes sense on top of the all-landmark survival term.
	if all_landmark_rank and not all_landmark_nll:
		raise ValueError("all_landmark_rank=True requires all_landmark_nll=True")

	for i in tqdm(range(n_iter)):
		# Re-shuffle every epoch so batch composition varies across epochs.
		perm = torch.randperm(n_train)
		x_ep = x_train[perm]
		t_ep = t_train[perm]
		e_ep = e_train[perm]
		s_ep = s_train[perm] if s_train is not None else None
		s_all_ep = s_all_train[perm] if s_all_train is not None else None

		if augment:
			visit_days_ep = visit_days_train[perm]
			x_ep, s_ep = _augment_landmark_truncation(x_ep, s_ep, visit_days_ep, split_time_tensor, landmark_augment_prob)

		model.train()
		for j in range(0, n_train, bs):
			xb = x_ep[j : j + bs]
			tb = t_ep[j : j + bs]
			eb = e_ep[j : j + bs]
			sb = s_ep[j : j + bs] if s_ep is not None else None
			s_all_b = s_all_ep[j : j + bs] if s_all_ep is not None else None

			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
				if sb is not None:
					sb = sb.cuda()
				if s_all_b is not None:
					s_all_b = s_all_b.cuda()

			optimizer.zero_grad()
			if all_landmark_nll:
				all_landmark_objective(model, xb, tb, eb, sb, s_all_b,
									   alpha, beta, sigma, cause_weights,
									   chunk_rows = landmark_chunk_rows, backward = True,
									   rank_on_landmarks = all_landmark_rank)
			else:
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
			s_all_b = s_all_valid

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
				if sb is not None:
					sb = sb.cuda()
				if s_all_b is not None:
					s_all_b = s_all_b.cuda()

			# Early stopping must track the objective being optimised, so the
			# validation loss uses the same all-landmark form when enabled.
			if all_landmark_nll and s_all_b is not None:
				valid_loss = all_landmark_objective(
					model, xb, tb, eb, sb, s_all_b,
					alpha, beta, sigma, cause_weights,
					chunk_rows = landmark_chunk_rows, backward = False,
					rank_on_landmarks = all_landmark_rank,
				)
			else:
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

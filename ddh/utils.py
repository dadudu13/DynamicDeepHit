from ddh.losses import (total_loss, build_landmark_prefixes, landmark_nll_loss,
						censoring_survival_by_bin, landmark_row_days)
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

def all_landmark_objective(model, xb, tb, eb, sb, s_all_b,
						   alpha, beta, sigma, cause_weights,
						   chunk_rows = 512, backward = False,
						   visit_days_b = None, landmark_decay_days = None,
						   entry_weight_b = None,
						   gamma = 0.0, horizon_spec = None,
						   component_totals = None):
	"""
		Total loss with the survival term evaluated at EVERY landmark instead of
		only the final visit:

			L = (1-alpha-beta) * L_long + alpha * L_rank
				+ beta * (1/K) sum_i (1/V_i) sum_v -log P(T_i,K_i | T_i>s_iv, X_i,<=v)

		With gamma > 0 a horizon-specific binary likelihood is added on the same
		expanded rows (see ddh.losses.horizon_likelihood_loss):

			+ gamma * (1/K) (1/H) sum_h sum_i (1/V_i) sum_v
					w_ipcw * -log P_hat(status at day_s+h | T>day_s, X_i,<=v)

		It is ADDED, not carved out of alpha/beta, so gamma=0 is bit-identical to
		the objective every existing arm was trained under.

		Longitudinal and ranking losses stay on the full-history rows, so their
		tuned weights keep the same scale as the final-visit objective and only
		the beta term changes.  Expanding P patients into sum_i V_i prefix rows
		multiplies activation memory by the mean visit count, so the expanded
		rows are processed in chunks of `chunk_rows`; back-propagating each chunk
		separately is exact because the objective is a plain weighted sum over
		rows (gradients accumulate in .grad until the caller steps).

		Returns the scalar loss value.  With backward=True the caller is
		responsible for optimizer.zero_grad()/step() around the call.
	"""
	main = total_loss(model, xb, tb, eb, alpha, beta, sigma,
					  s = sb, cause_weights = cause_weights, include_nll = False)
	if backward:
		main.backward()
	total = main.item()

	x_exp, s_exp, patient_idx, weight = build_landmark_prefixes(
		xb, s_all_b, visit_days = visit_days_b, decay_days = landmark_decay_days,
		entry_weight = entry_weight_b)
	t_exp, e_exp = tb.long()[patient_idx], eb.int()[patient_idx]

	# The horizon term needs each row's landmark DAY (to locate the bin holding
	# day_s + h); the NLL only ever needed its landmark BIN.
	day_exp = None
	use_horizon = gamma != 0 and horizon_spec is not None and visit_days_b is not None
	if use_horizon:
		day_exp = landmark_row_days(x_exp, patient_idx, visit_days_b)

	n_rows = x_exp.shape[0]
	step = n_rows if not chunk_rows or chunk_rows <= 0 else chunk_rows
	for start in range(0, n_rows, step):
		sl = slice(start, start + step)
		spec = None
		if use_horizon:
			spec = dict(horizon_spec)
			spec["day"] = day_exp[sl]
		parts = {} if component_totals is not None else None
		nll = landmark_nll_loss(model, x_exp[sl], t_exp[sl], e_exp[sl],
								s_exp[sl], weight[sl], nll_weight = beta,
								horizon_spec = spec, gamma = gamma,
								components = parts)
		if backward:
			nll.backward()
		total += nll.item()
		if parts is not None:
			# Raw, unweighted magnitudes -- what gamma has to be set against.
			component_totals["nll"] = component_totals.get("nll", 0.0) + parts["nll"]
			component_totals["horizon"] = component_totals.get("horizon", 0.0) + parts["horizon"]

	return total

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
			  visit_days_valid = None,
			  landmark_decay_days = None,
			  entry_weight_train = None,
			  entry_weight_valid = None,
			  gamma = 0.0,
			  horizon_days = None,
			  horizon_ipcw_clip = 10.0,
			  event_of_interest = 0):

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

	# ---- horizon-specific likelihood term -------------------------------------
	# The censoring distribution is estimated ONCE, from this fold's training rows
	# only, and held fixed: re-fitting it per batch would make the weights noisy
	# and would let batch composition move the objective.
	horizon_spec = None
	if gamma != 0:
		if not all_landmark_nll:
			raise ValueError("gamma > 0 requires all_landmark_nll=True")
		if split_time is None:
			raise ValueError("gamma > 0 requires split_time")
		if not horizon_days:
			raise ValueError("gamma > 0 requires horizon_days")
		n_bins = len(split_time) - 1
		g_bins = censoring_survival_by_bin(t_train, e_train, n_bins)
		horizon_spec = {
			"split_time": torch.as_tensor(split_time).double(),
			"horizons": [float(h) for h in horizon_days],
			"g_bins": g_bins,
			"ipcw_clip": float(horizon_ipcw_clip),
			"event_of_interest": int(event_of_interest),
		}
		print(f"[horizon] gamma={gamma} horizons={list(horizon_days)} "
			  f"bins={n_bins} G(last)={float(g_bins[-1]):.3f} "
			  f"max 1/G={float((1.0 / g_bins).clamp(max=horizon_ipcw_clip).max()):.2f}",
			  flush = True)

	for i in tqdm(range(n_iter)):
		# Re-shuffle every epoch so batch composition varies across epochs.
		perm = torch.randperm(n_train)
		x_ep = x_train[perm]
		t_ep = t_train[perm]
		e_ep = e_train[perm]
		s_ep = s_train[perm] if s_train is not None else None
		s_all_ep = s_all_train[perm] if s_all_train is not None else None

		# The decay weighting needs per-visit days too, not just the augmenter.
		visit_days_ep = None
		if augment or landmark_decay_days is not None:
			visit_days_ep = visit_days_train[perm]
		entry_w_ep = entry_weight_train[perm] if entry_weight_train is not None else None
		if augment:
			x_ep, s_ep = _augment_landmark_truncation(x_ep, s_ep, visit_days_ep, split_time_tensor, landmark_augment_prob)

		model.train()
		# Raw magnitudes of the two survival terms, so the gamma that makes them
		# comparable can be read off a run instead of guessed.
		epoch_parts = {} if (gamma != 0 and i < 3) else None
		clip_stats = [0, 0, 0.0] if i < 3 else None
		for j in range(0, n_train, bs):
			xb = x_ep[j : j + bs]
			tb = t_ep[j : j + bs]
			eb = e_ep[j : j + bs]
			sb = s_ep[j : j + bs] if s_ep is not None else None
			s_all_b = s_all_ep[j : j + bs] if s_all_ep is not None else None
			visit_days_b = visit_days_ep[j : j + bs] if visit_days_ep is not None else None
			entry_w_b = entry_w_ep[j : j + bs] if entry_w_ep is not None else None

			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
				if sb is not None:
					sb = sb.cuda()
				if s_all_b is not None:
					s_all_b = s_all_b.cuda()
				if visit_days_b is not None:
					visit_days_b = visit_days_b.cuda()

			optimizer.zero_grad()
			if all_landmark_nll:
				all_landmark_objective(model, xb, tb, eb, sb, s_all_b,
									   alpha, beta, sigma, cause_weights,
									   chunk_rows = landmark_chunk_rows, backward = True,
									   visit_days_b = visit_days_b,
									   landmark_decay_days = landmark_decay_days,
									   entry_weight_b = entry_w_b,
									   gamma = gamma, horizon_spec = horizon_spec,
									   component_totals = epoch_parts)
			else:
				loss = total_loss(model, xb, tb, eb, alpha, beta, sigma, s=sb, cause_weights=cause_weights)
				loss.backward()
			if grad_clip_norm > 0:
				# The binary log likelihood's gradient is -w/p for an event row, so at
				# a low event rate p a single event contributes about 1/p times what a
				# non-event does; if that makes clipping bite far more often with gamma > 0,
				# the clip is rescaling the whole update in the events' favour and
				# the arm's over-prediction is an artefact of the clip, not of the
				# term.  Cheap to record, and it decides that question from the runs
				# already being done rather than from a separate experiment.
				gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
				if clip_stats is not None:
					clip_stats[0] += 1
					clip_stats[1] += int(float(gn) > grad_clip_norm)
					clip_stats[2] += float(gn)
			optimizer.step()

		# Validation — skip gradient tracking to save memory and compute.
		model.eval()
		with torch.no_grad():
			xb, tb, eb = x_valid, t_valid, e_valid
			sb = s_valid
			s_all_b = s_all_valid
			visit_days_b = visit_days_valid
			entry_w_b = entry_weight_valid

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
				if sb is not None:
					sb = sb.cuda()
				if s_all_b is not None:
					s_all_b = s_all_b.cuda()
				if visit_days_b is not None:
					visit_days_b = visit_days_b.cuda()

			# Early stopping must track the objective being optimised, so the
			# validation loss uses the same all-landmark form when enabled.
			if all_landmark_nll and s_all_b is not None:
				valid_loss = all_landmark_objective(
					model, xb, tb, eb, sb, s_all_b,
					alpha, beta, sigma, cause_weights,
					chunk_rows = landmark_chunk_rows, backward = False,
					visit_days_b = visit_days_b,
					landmark_decay_days = landmark_decay_days,
					entry_weight_b = entry_w_b,
					gamma = gamma, horizon_spec = horizon_spec,
				)
			else:
				valid_loss = total_loss(
					model, xb, tb, eb, alpha, beta, sigma,
					s=sb, cause_weights=cause_weights
				).item()
		if clip_stats and clip_stats[0]:
			print(f"[grad] epoch {i}: clipped on {clip_stats[1]}/{clip_stats[0]} batches "
				  f"({100 * clip_stats[1] / clip_stats[0]:.0f}%), mean pre-clip norm "
				  f"{clip_stats[2] / clip_stats[0]:.2f} (limit {grad_clip_norm})", flush = True)
		if epoch_parts:
			print(f"[horizon] epoch {i}: sum raw NLL {epoch_parts['nll']:.1f}  "
				  f"sum raw horizon {epoch_parts['horizon']:.1f}  "
				  f"contribution beta*NLL {beta * epoch_parts['nll']:.1f} vs "
				  f"gamma*horizon {gamma * epoch_parts['horizon']:.1f}", flush = True)
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

from ddh.dsm_compat import (
	DeepRecurrentSurvivalMachines,
	_get_padded_features,
	_get_padded_targets,
	_reshape_tensor_with_nans,
)
from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.utils import train_ddh
from ddh.losses import total_loss
import numpy as np
import torch
import random

last = lambda t: np.array([t_[-1] for t_ in t], dtype = float)

class DynamicDeepHit(DeepRecurrentSurvivalMachines):
	
	def __init__(self, split = 50, layers_rnn = 1, hidden_rnn = 10, typ = 'LSTM',
		long_param = {}, att_param = {}, cs_param = {},
		alpha = 0.1, beta = 0.1, sigma = 0.1, cuda = False,
		cancer_vocab_size = None, cancer_embedding_dim = None, cancer_id_col = None,
		cause_weights = None, hazard = False, proportional = False):

		if isinstance(split, int):
			self.split = split
			self.split_time = None
		else:
			self.split = len(split) - 1
			self.split_time = split

		self.layers_rnn = layers_rnn
		self.hidden_rnn = hidden_rnn
		self.typ = typ
		
		self.long_param = long_param
		self.att_param = att_param
		self.cs_param = cs_param

		self.alpha = alpha
		self.beta = beta
		self.sigma = sigma
		self.cancer_vocab_size = cancer_vocab_size
		self.cancer_embedding_dim = cancer_embedding_dim
		self.cancer_id_col = cancer_id_col
		self.cause_weights = cause_weights
		# Discrete-time cause-specific hazard output layer (see DynamicDeepHitTorch).
		# Default False keeps the original DeepHit softmax-over-(cause, bin) head.
		self.hazard = hazard
		# Proportional factorisation of the hazard head (see DynamicDeepHitTorch).
		self.proportional = proportional

		self.cuda = cuda
		self.fitted = False
		  
	def _gen_torch_model(self, inputdim, optimizer, risks):
		model = DynamicDeepHitTorch(inputdim, self.split, self.layers_rnn, self.hidden_rnn,
				long_param = self.long_param, att_param = self.att_param, cs_param = self.cs_param,
				typ = self.typ, optimizer = optimizer, risks = risks,
				cancer_vocab_size = self.cancer_vocab_size,
				cancer_embedding_dim = self.cancer_embedding_dim,
				cancer_id_col = self.cancer_id_col,
				hazard = self.hazard,
				proportional = self.proportional).double()
		if self.cuda:
			model = model.cuda()
		return model

	def cpu(self):
		self.cuda = False
		if self.torch_model:
			self.torch_model = self.torch_model.cpu()
		return self

	def fit(self, x, t, e, vsize = 0.15, val_data = None,
		  iters = 1, learning_rate = 1e-3, batch_size = 100,
		  optimizer = "Adam", random_state = 100, s = None, s_val = None,
		  patience_limit = 10, grad_clip_norm = 1.0,
		  visit_days = None, landmark_augment_prob = 0.0, landmark_decay_days = None,
		  landmark_row_weighting = "patient", entry_weight = None,
		  row_weights = None, row_weights_val = None, landmark_band_edges = None,
		  visit_days_val = None, all_landmark_nll = False,
		  landmark_chunk_rows = 512,
		  gamma = 0.0, horizon_days = None, horizon_ipcw_clip = 10.0,
		  event_of_interest = 0):
		random.seed(random_state)
		np.random.seed(random_state)
		torch.manual_seed(random_state)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(random_state)
			torch.cuda.manual_seed_all(random_state)
		discretized_t, self.split_time = self.discretize(t, self.split, self.split_time)

		# Discretize landmark times into bin indices now that split_time is known
		max_bin = len(self.split_time) - 2
		s_disc = None if s is None else np.clip(
			np.digitize(s, self.split_time, right=True) - 1, 0, max_bin
		)
		s_val_disc = None if s_val is None else np.clip(
			np.digitize(s_val, self.split_time, right=True) - 1, 0, max_bin
		)

		if all_landmark_nll and visit_days is None:
			raise ValueError("all_landmark_nll=True requires visit_days")

		processed_data = self._prepocess_training_data(
			x, last(discretized_t), last(e),
			vsize, val_data, random_state,
			s=s_disc, s_val=s_val_disc, visit_days=visit_days,
			visit_days_val=visit_days_val, entry_weight=entry_weight,
			row_weights=row_weights, row_weights_val=row_weights_val,
		)
		(x_train, t_train, e_train, x_val, t_val, e_val,
		 s_train_tensor, s_val_tensor,
		 visit_days_train_tensor, visit_days_val_tensor,
		 entry_w_train_tensor, entry_w_val_tensor,
		 row_w_train_tensor, row_w_val_tensor) = processed_data

		# Landmark bin of *every* timestep (not just the last observation), used
		# by the all-landmark survival objective.
		s_all_train = s_all_val = None
		if all_landmark_nll:
			s_all_train = self._visit_day_bins(visit_days_train_tensor)
			if visit_days_val_tensor is None:
				raise ValueError(
					"all_landmark_nll=True with external val_data requires visit_days_val"
				)
			s_all_val = self._visit_day_bins(visit_days_val_tensor)
		inputdim = x_train.shape[-1]

		maxrisk = int(np.nanmax(e_train.cpu().numpy()))

		model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk)
		model = train_ddh(model,
					x_train, t_train, e_train,
					x_val, t_val, e_val,
					self.alpha, self.beta, self.sigma,
					n_iter = iters,
					lr = learning_rate,
					bs = batch_size,
					cuda = self.cuda,
					s_train = s_train_tensor,
					s_valid = s_val_tensor,
					cause_weights = self.cause_weights,
					patience_limit = patience_limit,
					grad_clip_norm = grad_clip_norm,
					visit_days_train = visit_days_train_tensor,
					split_time = self.split_time,
					landmark_augment_prob = landmark_augment_prob,
					s_all_train = s_all_train,
					s_all_valid = s_all_val,
					all_landmark_nll = all_landmark_nll,
					landmark_chunk_rows = landmark_chunk_rows,
					visit_days_valid = visit_days_val_tensor,
					landmark_decay_days = landmark_decay_days,
					landmark_row_weighting = landmark_row_weighting,
					row_weights_train = row_w_train_tensor,
					row_weights_valid = row_w_val_tensor,
					landmark_band_edges = landmark_band_edges,
					entry_weight_train = entry_w_train_tensor,
					entry_weight_valid = entry_w_val_tensor,
					gamma = gamma,
					horizon_days = horizon_days,
					horizon_ipcw_clip = horizon_ipcw_clip,
					event_of_interest = event_of_interest)

		self.torch_model = model.eval()
		self.fitted = True
		self.cpu()

		return self

	def discretize(self, t, split, split_time = None):
		"""
			Discretize the survival horizon

			Args:
				t (List of Array): Time of events
				split (int): Number of bins
				split_time (List, optional): List of bins (must be same length than split). Defaults to None.

			Returns:
				List of Array: Disretized events time
		"""
		if split_time is None:
			_, split_time = np.histogram(np.concatenate(t), split - 1)
		max_bin = len(split_time) - 2
		t_discretized = np.array(
			[
				np.clip(np.digitize(t_, split_time, right = True) - 1, 0, max_bin)
				for t_ in t
			],
			dtype = object
		)
		return t_discretized, split_time

	def _visit_day_bins(self, padded_days):
		"""
			Map a padded [N, T] array of per-visit elapsed days to landmark bin
			indices, using the same digitisation as the per-patient `s` in fit()
			so a patient's last visit gets the identical bin either way.
			Padded (NaN) positions bin to 0 and are never indexed.
		"""
		days = padded_days.detach().cpu().numpy()
		max_bin = len(self.split_time) - 2
		bins = np.clip(
			np.digitize(np.nan_to_num(days, nan = 0.0), self.split_time, right = True) - 1,
			0, max_bin,
		)
		return torch.from_numpy(bins).long()

	def _prepocess_test_data(self, x):
		data = torch.from_numpy(_get_padded_features(x))
		if self.cuda:
			data = data.cuda()
		return data

	def _prepocess_training_data(self, x, t, e, vsize, val_data, random_state, s=None, s_val=None, visit_days=None, visit_days_val=None, entry_weight=None, row_weights=None, row_weights_val=None):
		"""RNNs require different preprocessing for variable length sequences"""

		idx = list(range(x.shape[0]))
		np.random.seed(random_state)
		np.random.shuffle(idx)

		x = _get_padded_features(x)
		x_train, t_train, e_train = x[idx], t[idx], e[idx]

		x_train = torch.from_numpy(x_train).double()
		t_train = torch.from_numpy(t_train).double()
		e_train = torch.from_numpy(e_train).double()

		s_train_tensor, s_val_tensor = None, None
		if s is not None:
			s_shuffled = s[np.array(idx)]
			s_train_tensor = torch.from_numpy(s_shuffled).long()

		# Per-timestep elapsed-day, used only to re-bin `s` when a sequence is
		# truncated for landmark augmentation during training (see train_ddh).
		visit_days_train_tensor, visit_days_val_tensor = None, None
		if visit_days is not None:
			visit_days_padded = _get_padded_targets(visit_days)[np.array(idx), :, 0]
			visit_days_train_tensor = torch.from_numpy(visit_days_padded).double()

		# Per-(patient, visit) loss weight, e.g. inverse-intensity weights.  Padded and
		# shuffled exactly like visit_days so row v of patient i keeps its own weight.
		row_w_train_tensor, row_w_val_tensor = None, None
		if row_weights is not None:
			row_w_padded = _get_padded_targets(row_weights)[np.array(idx), :, 0]
			row_w_train_tensor = torch.from_numpy(row_w_padded).double()

		# Per-patient between-patient weight; shuffled by the same idx so it stays
		# aligned with x/t/e through the internal train/val split below.
		entry_w_train_tensor, entry_w_val_tensor = None, None
		if entry_weight is not None:
			entry_w_train_tensor = torch.from_numpy(
				np.asarray(entry_weight, dtype = float)[np.array(idx)]).double()

		if val_data is None:

			vsize = int(vsize*x_train.shape[0])

			x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

			x_train = x_train[:-vsize]
			t_train = t_train[:-vsize]
			e_train = e_train[:-vsize]

			if s_train_tensor is not None:
				s_val_tensor = s_train_tensor[-vsize:]
				s_train_tensor = s_train_tensor[:-vsize]

			if visit_days_train_tensor is not None:
				visit_days_val_tensor = visit_days_train_tensor[-vsize:]
				visit_days_train_tensor = visit_days_train_tensor[:-vsize]

			if entry_w_train_tensor is not None:
				entry_w_val_tensor = entry_w_train_tensor[-vsize:]
				entry_w_train_tensor = entry_w_train_tensor[:-vsize]

			if row_w_train_tensor is not None:
				row_w_val_tensor = row_w_train_tensor[-vsize:]
				row_w_train_tensor = row_w_train_tensor[:-vsize]

		else:

			x_val, t_val, e_val = val_data

			x_val = _get_padded_features(x_val)
			t_val, _ = self.discretize(t_val, self.split, self.split_time)

			x_val = torch.from_numpy(x_val).double()
			t_val = torch.from_numpy(last(t_val)).double()
			e_val = torch.from_numpy(last(e_val)).double()

			if s_val is not None:
				s_val_tensor = torch.from_numpy(s_val).long()

			if visit_days_val is not None:
				visit_days_val_tensor = torch.from_numpy(
					_get_padded_targets(visit_days_val)[:, :, 0]
				).double()

			if row_weights_val is not None:
				row_w_val_tensor = torch.from_numpy(
					_get_padded_targets(row_weights_val)[:, :, 0]
				).double()

		return (x_train, t_train, e_train,
				x_val, t_val, e_val,
				s_train_tensor, s_val_tensor,
				visit_days_train_tensor, visit_days_val_tensor,
				entry_w_train_tensor, entry_w_val_tensor,
				row_w_train_tensor, row_w_val_tensor)

	def compute_nll(self, x, t, e, s=None):
		if not self.fitted:
			raise Exception("The model has not been fitted yet. Please fit the " +
									"model using the `fit` method on some training data " +
									"before calling `_eval_nll`.")
		discretized_t, _ = self.discretize(t, self.split, self.split_time)
		max_bin = len(self.split_time) - 2
		s_disc = None if s is None else np.clip(
			np.digitize(s, self.split_time, right=True) - 1, 0, max_bin
		)
		processed_data = self._prepocess_training_data(
			x, last(discretized_t), last(e), 0, None, 0, s=s_disc
		)
		_, _, _, x_val, t_val, e_val, _, s_val, _, _ = processed_data
		return total_loss(self.torch_model, x_val, t_val, e_val, 0, 1, 1, s=s_val).item()

	def predict_survival(self, x, t, risk=1, all_step=False, fill_from_last=False, bs=100):
		"""Predict survival function for the given data.
		Args:
			x (List of Array): Input data, where each element is a sequence of features for a patient.
			t (Array): Time points at which to evaluate the survival function.
			risk (int, optional): Risk index to predict. Defaults to 1.
			all_step (bool, optional): If True, returns survival function for all time steps otherwise only last. 
				Defaults to False.
			fill_from_last (bool, optional): If True, return all steps from last time point to first. 
				Defaults to False (index 0 is the first observed time, otherwise last observed point). 
				Ignored if all_step is False.
			bs (int, optional): Batch size for prediction. Defaults to 100.
		Returns:
			Array(point, time_eval, (time_step)): Predicted survival function for each patient at the specified time points."""
		l = [len(x_) for x_ in x]

		if all_step:
			new_x = []
			for x_, l_ in zip(x, l):
				new_x += [x_[:li + 1] for li in range(l_)]
			x = new_x

		if not isinstance(t, list):
			t = [t]
		t = self.discretize([t], self.split, self.split_time)[0][0]

		if self.fitted:
			batches = int(len(x)/bs) + 1
			scores = {t_: [] for t_ in t}
			for j in range(batches):
				xb = self._prepocess_test_data(x[j*bs:(j+1)*bs])
				_, f = self.torch_model(xb)
				for t_ in t:
					scores[t_].append(torch.cumsum(f[int(risk) - 1], dim = 1)[:, t_].unsqueeze(1).detach().numpy())
			predictions = 1 - np.concatenate([np.concatenate(scores[t_], axis = 0) for t_ in t], axis = 1)
			if all_step:
				results = np.full((len(l), len(t), max(l)), float('nan'))
				start = 0
				for i, li in enumerate(l):
					if fill_from_last:
						# Reverse to have index 0 containing last observed time
						results[i, :, :li] = predictions[start:start+li][::-1].T
					else:
						results[i, :, :li] = predictions[start:start+li].T
					start += li
				return results
			else:
				return predictions
		else:
			raise Exception("The model has not been fitted yet. Please fit the " +
							"model using the `fit` method on some training data " +
							"before calling `predict_survival`.")

	def predict_risk(self, x, t, **args):
		if self.fitted:
			return 1-self.predict_survival(x, t, **args)
		else:
			raise Exception("The model has not been fitted yet. Please fit the " +
							"model using the `fit` method on some training data " +
							"before calling `predict_risk`.")

import torch.nn as nn
import torch
import numpy as np
from .utils import create_nn

class DynamicDeepHitTorch(nn.Module):

	def __init__(self, input_dim, output_dim, layers_rnn,
				hidden_rnn, long_param = {}, att_param = {}, cs_param = {},
				typ = 'LSTM', optimizer = 'Adam', risks = 1,
				cancer_vocab_size = None, cancer_embedding_dim = None,
				cancer_id_col = None, hazard = False, hazard_init_survival = 0.5):
		super(DynamicDeepHitTorch, self).__init__()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.optimizer = optimizer
		self.risks = risks
		self.typ = typ
		# hazard=False reproduces the original DeepHit output layer exactly: one
		# softmax over all (cause, bin) cells, which forces the total probability
		# of an event inside the grid to 1.  hazard=True instead emits per-bin
		# cause-specific hazards and converts them to the same (cause, bin)
		# probability interface, so every downstream consumer is unchanged while
		# the sum-to-one-over-the-horizon constraint is lifted.  See forward().
		self.hazard = hazard
		self.cancer_vocab_size = cancer_vocab_size
		self.cancer_embedding_dim = cancer_embedding_dim
		self.cancer_id_col = cancer_id_col
		self.use_cancer_embedding = (
			cancer_vocab_size is not None and
			cancer_embedding_dim is not None and
			cancer_id_col is not None
		)

		if self.use_cancer_embedding:
			self.cancer_embedding = nn.Embedding(cancer_vocab_size, cancer_embedding_dim)
			self.model_input_dim = input_dim - 1 + cancer_embedding_dim
		else:
			self.cancer_embedding = None
			self.model_input_dim = input_dim

		# RNN model for longitudinal data
		if self.typ == 'LSTM':
			self.embedding = nn.LSTM(self.model_input_dim, hidden_rnn, layers_rnn,
								batch_first=True)
		if self.typ == 'RNN':
			self.embedding = nn.RNN(self.model_input_dim, hidden_rnn, layers_rnn,
								batch_first=True, nonlinearity='relu')
		if self.typ == 'GRU':
			self.embedding = nn.GRU(self.model_input_dim, hidden_rnn, layers_rnn,
								batch_first=True)

		# Longitudinal network — exclude cancer_id_col from target when embedding is active
		longitudinal_output_dim = input_dim - 1 if self.use_cancer_embedding else input_dim
		self.longitudinal = create_nn(hidden_rnn, longitudinal_output_dim, no_activation_last = True, **long_param)

		# Attention mechanism
		self.attention = create_nn(self.model_input_dim + hidden_rnn, 1, no_activation_last = True, **att_param)
		self.attention_soft = nn.Softmax(1) # On temporal dimension

		# Cause specific network
		self.cause_specific = []
		for r in range(self.risks):
			self.cause_specific.append(create_nn(self.model_input_dim + hidden_rnn, output_dim, no_activation_last = True, **cs_param))
		self.cause_specific = nn.ModuleList(self.cause_specific)

		# Hazard mode adds one more head of the same shape, emitting the "no event
		# in this bin" logit.  Keeping the cause-specific heads exactly as they are
		# means the arm isolates the parameterisation change rather than confounding
		# it with a change in capacity or layer shapes.
		self.survive = (
			create_nn(self.model_input_dim + hidden_rnn, output_dim,
					  no_activation_last = True, **cs_param)
			if self.hazard else None
		)

		if self.hazard:
			# Without this the arm is unrunnable.  A softmax over K+1 outcomes puts
			# ~1/(K+1) on every hazard at init, so the total per-bin hazard starts
			# near K/(K+1) and survival collapses geometrically: with K=2 and 102
			# bins, S(last) ~ 5e-17 before a single gradient step.  The model would
			# spend its whole budget climbing out of "everyone fails immediately".
			#
			# Bias the survive logit so initial survival past the last bin is
			# hazard_init_survival.  With logits [0, .., 0, c] the per-cause hazard
			# is 1/(K + e^c), so solving S = (1 - K/(K + e^c))^B for c gives:
			lam_total = 1.0 - hazard_init_survival ** (1.0 / output_dim)
			bias = float(np.log(risks * (1.0 - lam_total) / lam_total))
			final_layer = self.survive[-1]
			nn.init.zeros_(final_layer.weight)
			nn.init.constant_(final_layer.bias, bias)

		# Probability
		self.soft = nn.Softmax(dim = -1) # On all observed output
		self.bin_soft = nn.Softmax(dim = -1) # Within each bin, over K+1 outcomes

	def forward(self, x):
		"""
			The forward function that is called when data is passed through DynamicDeepHit.
		"""
		if x.is_cuda:
			device = x.get_device()
		else:
			device = torch.device("cpu")

		# RNN representation - Nan values for not observed data
		x = x.clone()
		inputmask = torch.isnan(x[:, :, 0])
		x[inputmask] = 0

		if self.use_cancer_embedding:
			cancer_ids = x[:, :, self.cancer_id_col].long().clamp(0, self.cancer_vocab_size - 1)
			embedded_cancer = self.cancer_embedding(cancer_ids)
			embedded_cancer[inputmask] = 0
			x_without_cancer = torch.cat(
				[x[:, :, :self.cancer_id_col], x[:, :, self.cancer_id_col + 1:]],
				dim=2
			)
			x_model = torch.cat([x_without_cancer, embedded_cancer], dim=2)
		else:
			x_model = x

		hidden, _ = self.embedding(x_model)		
		
		# Longitudinal modelling
		longitudinal_prediction = self.longitudinal(hidden)

		# Attention using last observation to predict weight of all previously observed
		## Extract last observation (the one used for predictions)
		last_observations = ((~inputmask).sum(axis = 1) - 1)
		last_observations_idx = last_observations.unsqueeze(1).repeat(1, x.size(1))
		index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

		last = index == last_observations_idx
		x_last = x_model[last]

		## Concatenate all previous with new to measure attention
		concatenation = torch.cat([hidden, x_last.unsqueeze(1).repeat(1, x.size(1), 1)], -1)

		## Compute attention and normalize
		attention = self.attention(concatenation).squeeze(-1)
		attention[index >= last_observations_idx] = -1e10 # Want soft max to be zero as values not observed
		attention[last_observations > 0] = self.attention_soft(attention[last_observations > 0]) # Weight previous observation
		attention[last_observations == 0] = 0 # No context for only one observation

		# Risk networks
		# The original paper is not clear on how the last observation is
		# combined with the temporal sum, other code was concatenating them
		outcomes = []
		attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
		hidden_attentive = torch.sum(attention * hidden, axis = 1)
		hidden_attentive = torch.cat([hidden_attentive, x_last], 1)
		for cs_nn in self.cause_specific:
			outcomes.append(cs_nn(hidden_attentive))

		if self.hazard:
			# Per-bin competing-risks hazards.  Stack the K cause logits with the
			# "survive this bin" logit and softmax *within each bin* over K+1
			# outcomes, so lam[..., k] = P(fail in bin b from cause k | at risk at b)
			# and lam[..., K] = 1 - sum_k lam[..., k].
			logits = torch.stack(outcomes + [self.survive(hidden_attentive)], dim = -1)
			lam = self.bin_soft(logits)                       # [N, B, K+1]
			hazards = lam[..., :self.risks]                   # [N, B, K]
			step_survival = lam[..., self.risks]              # [N, B]

			# S(b-1) = prod_{b' < b} (1 - sum_k lam_k(b')), with S(-1) = 1.
			ones = torch.ones_like(step_survival[:, :1])
			survival_before = torch.cumprod(
				torch.cat([ones, step_survival[:, :-1]], dim = 1), dim = 1
			)                                                 # [N, B]

			# Convert to the same (cause, bin) probability interface the softmax
			# head produces: P(T = b, K = k) = lam_k(b) * S(b-1).  Cumulative sums
			# of these are the cause-specific CIFs and 1 - sum_k F_k(b) is S(b),
			# exactly as before -- but the total no longer has to reach 1 inside
			# the grid, because the leftover is P(no event by the last bin).
			pmf = hazards * survival_before.unsqueeze(-1)     # [N, B, K]
			return longitudinal_prediction, [pmf[:, :, k] for k in range(self.risks)]

		# Soft max for probability distribution
		outcomes = torch.cat(outcomes, dim = 1)
		outcomes = self.soft(outcomes)

		outcomes = [outcomes[:, i * self.output_dim : (i+1) * self.output_dim] for i in range(self.risks)]
		return longitudinal_prediction, outcomes
